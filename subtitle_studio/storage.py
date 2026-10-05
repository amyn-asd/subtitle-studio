from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .config import DATA, initialize
from .types import TRANSLATION_VERSION


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class Store:
    def __init__(self, path: Path | None = None):
        initialize()
        self.path = path or DATA / "studio.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, media TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, project_id TEXT NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS cues(id TEXT PRIMARY KEY, project_id TEXT NOT NULL, track INTEGER NOT NULL, start REAL NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS cue_project ON cues(project_id,track,start);
                CREATE TABLE IF NOT EXISTS translations(project_id TEXT NOT NULL, language TEXT NOT NULL, cue_id TEXT NOT NULL, text TEXT NOT NULL, source_text TEXT NOT NULL, PRIMARY KEY(project_id,language,cue_id));
                CREATE TABLE IF NOT EXISTS exports(project_id TEXT NOT NULL, track INTEGER NOT NULL, language TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(project_id,track,language));
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(translations)")}
            if "version" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
            if "model_digest" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN model_digest TEXT NOT NULL DEFAULT ''")
            if "source_signature" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN source_signature TEXT NOT NULL DEFAULT ''")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def create_project(self, media: dict) -> dict:
        with self.connect() as db:
            existing = db.execute("SELECT id FROM projects WHERE json_extract(media,'$.fingerprint')=?", (media["fingerprint"],)).fetchone()
            if existing:
                return self.project(existing["id"])
            pid = uuid.uuid4().hex
            db.execute("INSERT INTO projects VALUES(?,?,?)", (pid, encode(media), time.time()))
        return self.project(pid)

    def project(self, pid: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
            if not row:
                raise KeyError("Project not found")
            jobs = [json.loads(r[0]) for r in db.execute("SELECT body FROM jobs WHERE project_id=? ORDER BY updated DESC", (pid,))]
            counts = [dict(r) for r in db.execute("SELECT track,COUNT(*) AS count FROM cues WHERE project_id=? GROUP BY track", (pid,))]
        return {"id": pid, "media": json.loads(row["media"]), "created": row["created"], "jobs": jobs, "cue_counts": counts}

    def projects(self) -> list[dict]:
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM projects ORDER BY created DESC LIMIT 100")]
        return [self.project(pid) for pid in ids]

    def update_media(self, pid: str, media: dict):
        with self.connect() as db:
            db.execute("UPDATE projects SET media=? WHERE id=?", (encode(media), pid))

    def save_job(self, job: dict):
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,updated=excluded.updated",
                       (job["id"], job["project_id"], encode(job), time.time()))

    def job(self, jid: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            raise KeyError("Job not found")
        return json.loads(row[0])

    def recover_jobs(self):
        with self.connect() as db:
            rows = [json.loads(r[0]) for r in db.execute("SELECT body FROM jobs")]
        for job in rows:
            if job["status"] in ("running", "queued", "pausing"):
                job.update(status="paused", message="Interrupted when the app closed. Resume to continue from checkpoints.")
                self.save_job(job)

    def cues(self, pid: str, track: int | None = None) -> list[dict]:
        query = "SELECT body FROM cues WHERE project_id=?"
        values: list = [pid]
        if track is not None:
            query += " AND track=?"
            values.append(track)
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(query + " ORDER BY track,start", values)]

    def replace_cues(self, pid: str, track: int, cues: list[dict]):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Preserve user edits when rerunning the same cached project.
            previous = {c["id"]: c for c in (json.loads(r[0]) for r in db.execute("SELECT body FROM cues WHERE project_id=? AND track=?", (pid, track)))}
            db.execute("DELETE FROM cues WHERE project_id=? AND track=?", (pid, track))
            for cue in cues:
                old = previous.get(cue["id"], {})
                if old.get("edited") or old.get("reviewed"):
                    cue.update({k: old[k] for k in ("text", "start", "end", "reviewed", "edited") if k in old})
                db.execute("INSERT INTO cues VALUES(?,?,?,?,?)", (cue["id"], pid, track, cue["start"], encode(cue)))

    def update_cue(self, pid: str, cid: str, changes: dict, *, automated=False, expected_text=None) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM cues WHERE project_id=? AND id=?", (pid, cid)).fetchone()
            if not row:
                raise KeyError("Cue not found")
            cue = json.loads(row[0])
            # Human edits win even when an AI request was already in flight.
            if automated and (cue.get("edited") or cue.get("reviewed") or cue["text"] != expected_text):
                return cue
            cue.update(changes)
            if not cue["text"].strip() or cue["end"] <= cue["start"]:
                raise ValueError("A subtitle needs text and an end time after its start time.")
            db.execute("UPDATE cues SET body=?,start=? WHERE id=?", (encode(cue), cue["start"], cid))
        return cue

    def save_translation(self, pid: str, lang: str, cue: dict, text: str, *, version=TRANSLATION_VERSION, model_digest="", source_signature=None):
        if source_signature is None:
            from .translation import source_signature as signature
            source_signature = signature([cue], lang, model_digest)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO translations(project_id,language,cue_id,text,source_text,version,model_digest,source_signature) VALUES(?,?,?,?,?,?,?,?)",
                       (pid, lang, cue["id"], text, cue["text"], version, model_digest, source_signature))

    def translations(self, pid: str, lang: str) -> dict[str, dict]:
        with self.connect() as db:
            return {r["cue_id"]: dict(r) for r in db.execute("SELECT * FROM translations WHERE project_id=? AND language=?", (pid, lang))}

    def set_export(self, pid: str, track: int, lang: str, path: str):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO exports VALUES(?,?,?,?)", (pid, track, lang, path))
