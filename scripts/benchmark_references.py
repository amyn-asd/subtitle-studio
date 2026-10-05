"""Small reproducible reference benchmark through the application's GPU queue."""
from __future__ import annotations

import argparse
import json
import math
import time
import unicodedata
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from subtitle_studio.config import DATA


def units(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "fa":
        text = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    text = "".join(c if unicodedata.category(c)[0] in "LN" or c.isspace() else " " for c in text)
    return [c for c in text if not c.isspace()] if language == "ja" else text.split()


def distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, actual in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def bundle(rows: list[dict], path: Path, noise_db=None):
    recordings = []
    for row in rows:
        audio, rate = sf.read(row["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        if noise_db is not None:
            # Apply the requested SNR per recording: their recording levels vary considerably.
            noise = np.random.default_rng(42 + len(recordings)).standard_normal(len(audio)).astype(np.float32)
            noise *= np.sqrt(np.mean(audio * audio) / max(1e-8, np.mean(noise * noise))) / (10 ** (noise_db / 20))
            audio += noise
        recordings.extend([audio, np.zeros(16000, dtype=np.float32)])
    audio = np.concatenate(recordings)
    audio /= max(1, float(np.max(np.abs(audio))))
    sf.write(path, audio, 16000)


def wait(client, jid):
    last = None
    while True:
        response = client.get(f"/jobs/{jid}")
        response.raise_for_status()
        job = response.json()
        current = job["stage"], job["status"]
        if current != last:
            print(f"Benchmark: {current[0]} / {current[1]}", flush=True)
            last = current
        if job["status"] == "done":
            return job
        if job["status"] in ("failed", "paused", "cancelled"):
            raise RuntimeError(job.get("error", job["message"]))
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-db", type=float, help="Also benchmark seeded white noise at this SNR")
    parser.add_argument("--languages", nargs="+", help="Limit the benchmark to language codes, such as en fa ja")
    parser.add_argument("--recognition-only", action="store_true", help="Measure primary recognition without rechecks/discussion")
    args = parser.parse_args()
    rows = json.loads((DATA / "references" / "references.json").read_text(encoding="utf-8"))
    server = json.loads((DATA / "server.json").read_text())
    client = httpx.Client(base_url=f"http://127.0.0.1:{server['port']}/api",
                         headers={"X-Studio-Token": server["token"]}, timeout=300)
    results = []
    report = DATA / "validation" / ("reference-recognition-benchmark.json" if args.recognition_only else "reference-benchmark.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    for snr in [None] + ([args.noise_db] if args.noise_db is not None else []):
        for language in sorted({r["language"] for r in rows}):
            if args.languages and language not in args.languages:
                continue
            group = [r for r in rows if r["language"] == language]
            path = DATA / "references" / f"benchmark-{language}-{'clean' if snr is None else str(snr)+'db'}.wav"
            bundle(group, path, snr)
            response = client.post("/projects", json={"path": str(path)})
            response.raise_for_status()
            project = response.json()
            for job in project["jobs"]:
                if job["kind"] == "scan" and job["status"] in ("running", "queued"):
                    wait(client, job["id"])
            started = time.monotonic()
            response = client.post(f"/projects/{project['id']}/jobs", json={
                "tracks": [t["stream_index"] for t in project["media"]["audio_tracks"]],
                "settings": {"preset": "accurate", "recheck": not args.recognition_only, "debate": not args.recognition_only}})
            response.raise_for_status()
            wait(client, response.json()["id"])
            cues = client.get(f"/projects/{project['id']}/cues").json()
            reference = units(" ".join(r["reference"] for r in group), language)
            primary = units(" ".join(c["raw_text"] for c in cues), language)
            reviewed = units(" ".join(c["text"] for c in cues), language)
            result = {"language": language, "clips": len(group), "noise_snr_db": snr,
                      "metric": "CER" if language == "ja" else "WER", "reference_units": len(reference),
                      "primary_errors": distance(reference, primary), "reviewed_errors": distance(reference, reviewed),
                      "detected_languages": sorted({c["language"] for c in cues}),
                      "cue_count": len(cues), "flagged": sum(bool(c["flags"]) for c in cues),
                      "debates": sum(bool(c.get("decision")) for c in cues),
                      "wall_seconds": round(time.monotonic() - started, 2), "project_id": project["id"]}
            for key in ("primary", "reviewed"):
                result[key + "_error_rate"] = round(result[key + "_errors"] / max(1, len(reference)), 4)
            results.append(result)
            report.write_text(json.dumps({"source": "google/fleurs test", "scope": "Three clips per language; small diagnostic sample",
                                          "recognition_only": args.recognition_only,
                                          "results": results}, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
