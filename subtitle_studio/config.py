from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SUBTITLE_STUDIO_DATA", ROOT / "data"))
PREFERENCES = ROOT / "studio-config.json"
preferences = json.loads(PREFERENCES.read_text(encoding="utf-8-sig")) if PREFERENCES.exists() else {}
LOCAL_AI = Path(preferences["local_ai_root"]) if preferences.get("local_ai_root") else None
DEFAULT_MODELS = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "SubtitleStudio" / "models"
MODELS = Path(os.environ.get("SUBTITLE_STUDIO_MODELS", preferences.get("models_root", str(DEFAULT_MODELS))))
OLLAMA_DIRECTORY = Path(preferences.get("ollama_models", str(MODELS / "ollama")))
WEB = ROOT / "web"


def initialize() -> None:
    for path in (DATA, MODELS, DATA / "projects", DATA / "clips", DATA / "logs"):
        path.mkdir(parents=True, exist_ok=True)


def binary(name: str) -> str:
    configured = preferences.get("tools", {}).get(name)
    if configured and Path(configured).is_file():
        return configured
    local = ROOT / ".runtime" / "bin" / (name + ".exe" if os.name == "nt" else name)
    found = str(local) if local.exists() else shutil.which(name)
    if not found and name == "vlc":
        for base in (os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")):
            candidate = Path(base) / "VideoLAN" / "VLC" / "vlc.exe"
            if candidate.is_file():
                found = str(candidate)
                break
    if not found and name == "ollama":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            found = str(candidate)
    if not found:
        raise FileNotFoundError(f"{name} is not installed. Run Setup.ps1 or select its executable in Settings.")
    return found


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"HF_HOME": str(MODELS / "hf-cache"), "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "TORCH_HOME": str(MODELS / "torch"), "OMP_NUM_THREADS": "4"})
    return env
