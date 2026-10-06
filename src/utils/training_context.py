"""Safe checkpoint identity and RNG state, using tensors and primitives only."""
import hashlib
import random

import numpy as np
import torch


def rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [state[0], torch.tensor(state[1].astype("int64")),
                      int(state[2]), int(state[3]), float(state[4])]}


def restore_rng(state):
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, keys.cpu().numpy().astype("uint32"),
                             int(pos), int(has_gauss), float(cached)))


def data_fingerprint(manager):
    content = getattr(manager, "content_fingerprint", None)
    if content is not None:
        return str(content)
    digest = hashlib.sha256()
    raw = getattr(manager, "raw_dict", None) or {}
    values = dict(raw)
    # Synthetic managers may expose only features/returns; include those too.
    if not all(k in raw for k in ("open", "high", "low", "close")):
        for name in ("feat_tensor", "target_ret"):
            value = getattr(manager, name, None)
            if value is not None:
                values[name] = value
    for name in sorted(values):
        tensor = torch.as_tensor(values[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
