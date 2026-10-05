from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT, preferences

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".ts", ".mts"}


def locations() -> list[dict]:
    choices = [("Video folder", preferences.get("input_directory", ROOT.parent)), ("Home", Path.home())]
    if os.name == "nt":
        choices.extend((drive, drive) for drive in os.listdrives())
    else:
        choices.append(("Filesystem", Path("/")))
    seen, result = set(), []
    for name, raw in choices:
        path = Path(raw).expanduser()
        try:
            if not path.is_dir():
                continue
            path = path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(path))
        if key not in seen:
            result.append({"name": name, "path": str(path)})
            seen.add(key)
    return result


def list_files(path: str | None = None, *, kind="file", query="", offset=0, limit=200, files_only=False) -> dict:
    shortcuts = locations()
    directory = Path(path.strip().strip('"')).expanduser() if path else Path(shortcuts[0]["path"] if shortcuts else Path.home())
    try:
        if directory.is_file():
            directory = directory.parent
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Choose an existing folder")
        entries, skipped = [], 0
        with os.scandir(directory) as items:
            for item in items:
                if query.casefold() not in item.name.casefold():
                    continue
                try:
                    folder = item.is_dir()
                    if folder and files_only:
                        continue
                    if not folder and (kind == "folder" or Path(item.name).suffix.lower() not in MEDIA_EXTENSIONS or not item.is_file()):
                        continue
                    entries.append({"name": item.name, "path": str(directory / item.name), "kind": "folder" if folder else "file",
                                    "size": None if folder else item.stat().st_size})
                except OSError:
                    skipped += 1
    except FileNotFoundError as exc:
        raise ValueError("That folder does not exist. Check the path or choose a location.") from exc
    except PermissionError as exc:
        raise ValueError("This folder is not readable. Choose another folder.") from exc
    except OSError as exc:
        raise ValueError("This location is unavailable. Choose another folder.") from exc
    entries.sort(key=lambda entry: (entry["kind"] != "folder", entry["name"].casefold(), entry["name"]))
    return {"path": str(directory), "parent": str(directory.parent) if directory.parent != directory else None,
            "locations": shortcuts, "entries": entries[offset:offset + limit], "total": len(entries), "skipped": skipped}
