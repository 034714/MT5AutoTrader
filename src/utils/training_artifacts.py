"""Root-anchored artifact paths and same-directory atomic replacement."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile


def root_path(*parts) -> Path:
    import config
    return Path(getattr(config.Config, "ROOT_DIR", config.ROOT_DIR)).resolve().joinpath(*parts)


def checkpoint_dir() -> Path:
    from config import Config
    path = Path(Config.CHECKPOINT_DIR)
    return path if path.is_absolute() else root_path(path)


def strategy_path(symbol=None, timeframe=None) -> Path:
    from config import Config
    if symbol:
        tag = f"{symbol}_{timeframe}" if timeframe else symbol
        return root_path("strategies", f"best_{tag}.json")
    path = Path(Config.STRATEGY_FILE)
    return path if path.is_absolute() else root_path(path)


def history_path(tag=None) -> Path:
    name = f"training_history_{tag}.json" if tag else "training_history.json"
    return root_path("training_history", name)


def atomic_write(path, writer) -> Path:
    """Call writer(temp Path), then replace path; preserve old bytes on failure.

    Accepts any caller-selected path (including external caches). No path-security
    policy is imposed: validate untrusted input before calling this helper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_json_write(path, payload) -> Path:
    """Write strict UTF-8 JSON atomically; nonfinite values raise ValueError."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def json_safe(value):
    """Represent unavailable/nonfinite history metrics as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def checkpoint_step(path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", Path(path).name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(pattern) -> Path | None:
    paths = [p for p in checkpoint_dir().glob(pattern) if checkpoint_step(p) >= 0]
    return max(paths, key=checkpoint_step) if paths else None
