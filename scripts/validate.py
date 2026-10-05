"""Run real media through the running app; never uploads media or prints dialogue."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import httpx

from subtitle_studio.config import DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model", choices=["turbo", "large"], default="turbo")
    parser.add_argument("--original-audio", action="store_true", help="Disable the recommended audio enhancements")
    args = parser.parse_args()
    server = json.loads((DATA / "server.json").read_text())
    client = httpx.Client(base_url=f"http://127.0.0.1:{server['port']}/api", headers={"X-Studio-Token":server["token"]}, timeout=300)
    results = []
    for index, path in enumerate(args.videos):
        response = client.post("/projects", json={"path":str(path.resolve())})
        response.raise_for_status()
        project = response.json()
        tracks = [t["stream_index"] for t in project["media"]["audio_tracks"]]
        response = client.post(f"/projects/{project['id']}/jobs", json={"tracks":tracks,"settings":{
            "preset":"fast" if args.model == "turbo" else "accurate","enhance_audio":not args.original_audio,"start_seconds":0 if args.full else args.start,
            "limit_seconds":None if args.full else args.seconds}})
        response.raise_for_status()
        job = response.json()
        started = time.time()
        last = None
        peak = 0
        while True:
            job = client.get(f"/jobs/{job['id']}").json()
            state = (job["stage"], int(job["progress"] * 10), job["status"])
            if state != last:
                print(f"Media {index+1}: {job['stage']} {job['progress']:.0%} {job['status']}", flush=True)
                last = state
            try:
                queried = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3)
                peak = max(peak, int(queried.stdout.strip().splitlines()[0]))
            except (OSError, ValueError, IndexError):
                pass
            if job["status"] in ("done","failed","paused","cancelled"):
                break
            time.sleep(2)
        cues = client.get(f"/projects/{project['id']}/cues").json()
        export_paths = []
        if job["status"] == "done":
            for track in tracks:
                selected = [c for c in cues if c["track"] == track]
                if selected:
                    export = client.post(f"/projects/{project['id']}/export",json={"track":track}).json()
                    export_paths.append(export.get("path"))
        result = {"media_index":index+1,"project_id":project["id"],"source_duration":project["media"]["duration"],
                  "processed_seconds":project["media"]["duration"] if args.full else args.seconds,
                  "wall_seconds":round(time.time()-started,2),"status":job["status"],"error":job.get("error"),
                  "peak_gpu_total_mb":peak,"cue_count":len(cues),"languages":sorted({c["language"] for c in cues}),
                  "flagged":sum(bool(c["flags"]) for c in cues),
                  "manual_edits":sum(bool(c.get("edited")) for c in cues),"exports":export_paths,"warnings":job["warnings"]}
        results.append(result)
        output = DATA / "validation" / ("full-run.json" if args.full else "sample-run.json")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k not in ("exports","error","warnings")}),flush=True)
        if job["status"] != "done":
            print(job.get("error",job["message"]),flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
