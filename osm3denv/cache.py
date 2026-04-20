from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def bbox_key(bbox: tuple[float, float, float, float]) -> str:
    s = ",".join(f"{v:.6f}" for v in bbox)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def read_json(path: Path, max_age_s: float | None = None) -> dict | None:
    if not path.exists():
        return None
    if max_age_s is not None and (time.time() - path.stat().st_mtime) > max_age_s:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
