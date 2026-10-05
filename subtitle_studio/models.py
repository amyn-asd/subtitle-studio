from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .config import DATA, MODELS, LOCAL_AI, OLLAMA_DIRECTORY, preferences, binary, child_environment, initialize
from .media import NO_WINDOW
from .subtitles import atomic_text

HF_MODELS = {
    "whisper": {"name": "Whisper large-v3", "repo": "Systran/faster-whisper-large-v3", "required": "model.bin", "size_gb": 3.1},
    "turbo": {"name": "Whisper turbo", "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "required": "model.bin", "size_gb": 1.6},
    "qwen_asr": {"name": "Qwen3-ASR 1.7B", "repo": "Qwen/Qwen3-ASR-1.7B", "required": "config.json", "size_gb": 4.4},
    "lid": {"name": "107-language detector", "repo": "speechbrain/lang-id-voxlingua107-ecapa", "required": "embedding_model.ckpt", "size_gb": .1},
}
OLLAMA_MODELS = {"context": "qwen3.5:9b", "translation": "translategemma:12b"}


def model_path(key: str) -> Path:
    registry = MODELS / "registry.json"
    if registry.exists():
        entry = json.loads(registry.read_text(encoding="utf-8")).get(key)
        if entry and Path(entry["path"]).is_dir():
            return Path(entry["path"])
    return MODELS / key


def model_ready(key: str) -> bool:
    spec = HF_MODELS[key]
    directory = model_path(key)
    return (directory / spec["required"]).is_file() and model_revision(key) != "missing"


def model_revision(key: str) -> str:
    registry = MODELS / "registry.json"
    if registry.exists():
        entry = json.loads(registry.read_text(encoding="utf-8")).get(key)
        if entry:
            return entry["revision"]
    marker = model_path(key) / "studio-revision.json"
    return json.loads(marker.read_text(encoding="utf-8"))["revision"] if marker.exists() else "missing"


def discover_existing():
    """Register existing compatible snapshots in place, without modifying shared weights."""
    initialize()
    registry_path = MODELS / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    caches = [Path.home() / ".cache" / "huggingface" / "hub"]
    if os.environ.get("HF_HOME"):
        caches.insert(0, Path(os.environ["HF_HOME"]) / "hub")
    if LOCAL_AI:
        caches = [LOCAL_AI / "data" / "huggingface" / "hub", LOCAL_AI / "models" / "huggingface" / "hub"] + caches
    caches = [Path(p) for p in preferences.get("model_search_paths", [])] + caches
    for key, spec in HF_MODELS.items():
        if key in registry and (Path(registry[key]["path"]) / spec["required"]).is_file():
            continue
        for cache in caches:
            snapshots = cache / ("models--" + spec["repo"].replace("/", "--")) / "snapshots"
            if not snapshots.exists():
                continue
            candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in candidates:
                if (path / spec["required"]).is_file() and (path / spec["required"]).stat().st_size > (1000000 if key != "qwen_asr" else 100):
                    if key == "qwen_asr" and not list(path.glob("*.safetensors")):
                        continue
                    registry[key] = {"path": str(path), "revision": path.name, "reused": True}
                    break
            if key in registry:
                break
    atomic_text(registry_path, json.dumps(registry, indent=2))
    return registry


def download_hf(key: str, update):
    from huggingface_hub import HfApi, snapshot_download
    if model_ready(key):
        update(f"{HF_MODELS[key]['name']} already installed", 1)
        return
    spec = HF_MODELS[key]
    info = HfApi().model_info(spec["repo"])
    update(f"Downloading {spec['name']} ({spec['size_gb']} GB); downloads resume automatically", 0)
    path = model_path(key)
    snapshot_download(repo_id=spec["repo"], revision=info.sha, local_dir=path, max_workers=4,
                      allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.yaml", "*.ckpt", "*.pt"])
    atomic_text(path / "studio-revision.json", json.dumps({"repo": spec["repo"], "revision": info.sha}, indent=2))
    update(f"{spec['name']} installed", 1)


class Ollama:
    """An app-owned loopback server and model directory; never changes the user's Ollama server."""
    def __init__(self):
        self.process = None
        self.port = None
        self.lock = threading.Lock()
        self.log = None
        atexit.register(self.close)

    def start(self) -> str:
        with self.lock:
            if self.process and self.process.poll() is None:
                return self.url
            initialize()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
            env = child_environment()
            env.update(OLLAMA_HOST=f"127.0.0.1:{self.port}", OLLAMA_MODELS=str(OLLAMA_DIRECTORY),
                       OLLAMA_NO_CLOUD="1", OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1")
            self.log = (DATA / "logs" / "ollama.log").open("ab")
            self.process = subprocess.Popen([binary("ollama"), "serve"], env=env, stdout=self.log, stderr=self.log, creationflags=NO_WINDOW)
            for _ in range(150):
                if self.process.poll() is not None:
                    raise RuntimeError("Private Ollama could not start. See data/logs/ollama.log.")
                try:
                    httpx.get(self.url + "/api/version", timeout=.3).raise_for_status()
                    return self.url
                except (httpx.HTTPError, OSError):
                    time.sleep(.1)
            raise RuntimeError("Timed out starting private Ollama.")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def tags(self) -> list[dict]:
        if not self.process or self.process.poll() is not None:
            return []
        try:
            return httpx.get(self.url + "/api/tags", timeout=3).json().get("models", [])
        except httpx.HTTPError:
            return []

    def ready(self, key: str) -> bool:
        return any(m["name"] == OLLAMA_MODELS[key] for m in self.tags())

    def pull(self, key: str, update):
        self.start()
        if self.ready(key):
            update(f"{OLLAMA_MODELS[key]} already installed", 1)
            return
        with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as client:
            with client.stream("POST", self.url + "/api/pull", json={"model": OLLAMA_MODELS[key], "stream": True}) as response:
                response.raise_for_status()
                last_report = 0
                last_status = None
                for line in response.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if "error" in item:
                        raise RuntimeError(item["error"])
                    progress = item.get("completed", 0) / max(1, item.get("total", 1))
                    if time.monotonic() - last_report > .3 or item.get("status") != last_status:
                        update(f"{OLLAMA_MODELS[key]}: {item.get('status', 'Downloading')}", progress)
                        last_report, last_status = time.monotonic(), item.get("status")
        # Retain the actual digest used; pulls are explicit, never automatic upgrades.
        atomic_text(MODELS / f"{key}-revision.json", json.dumps(self.tags(), indent=2))

    def chat(self, model: str, messages: list[dict], schema: dict | None = None) -> str:
        self.start()
        payload = {"model": model, "messages": messages, "stream": False, "keep_alive": "10m",
                   "think": False, "options": {"temperature": 0,
                       "num_ctx": 2048 if model == OLLAMA_MODELS["translation"] else 8192, "num_predict": 768}}
        if schema:
            payload["format"] = schema
        with httpx.Client(timeout=240) as client:
            response = client.post(self.url + "/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
            if body.get("error"):
                raise RuntimeError(body["error"])
            return body["message"]["content"]

    def unload(self, model: str):
        if self.process and self.process.poll() is None:
            try:
                httpx.post(self.url + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=45).raise_for_status()
            except httpx.HTTPError:
                # Killing the app-owned server guarantees GPU release after an unload failure.
                self.close()

    def close(self):
        if self.process and self.process.poll() is None:
            # Unload this private server's runners before stopping it. In
            # particular, Windows child processes can outlive their parent.
            try:
                models = httpx.get(self.url + "/api/ps", timeout=3).json().get("models", [])
                for model in models:
                    httpx.post(self.url + "/api/generate", json={"model": model["name"], "keep_alive": 0}, timeout=15).raise_for_status()
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill.exe", "/PID", str(self.process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10, creationflags=NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.log:
            self.log.close()
            self.log = None


ollama = Ollama()


def status() -> list[dict]:
    installed = []
    for key, spec in HF_MODELS.items():
        installed.append({"id": key, "name": spec["name"], "ready": model_ready(key), "size_gb": spec["size_gb"]})
    tags = ollama.tags()
    for key, tag in OLLAMA_MODELS.items():
        installed.append({"id": key, "name": tag, "ready": any(m["name"] == tag for m in tags),
                          "size_gb": 7.6 if key == "context" else 8.1})
    return installed
