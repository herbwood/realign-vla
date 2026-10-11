"""Small FP16 feature shards; the optimization cache contains no policy model."""
import json
from pathlib import Path
import torch
from .utils import sha256


def load_feature_cache(directory: str | Path) -> dict:
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    clean, corrupted, source, split = [], [], [], []
    offset = 0
    for entry in index["shards"]:
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Feature shard must be inside the cache")
        path = directory / relative
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Feature shard content changed: {relative}")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        c, p, src = shard["clean"], shard["corrupted"], shard["source"]
        if c.ndim != 3 or c.shape[1:] != p.shape[1:] or p.ndim != 3:
            raise ValueError("Expected matched [frames,tokens,channels] feature tensors")
        if c.dtype != torch.float16 or p.dtype != torch.float16:
            raise ValueError("The paper feature cache uses FP16 storage")
        if src.ndim != 1 or len(src) != len(p) or src.dtype != torch.long or not len(c):
            raise ValueError("Invalid clean-frame source mapping")
        if src.min() < 0 or src.max() >= len(c) or entry["split"] not in ("train", "val"):
            raise ValueError("Invalid source indices or data split")
        clean.append(c); corrupted.append(p); source.append(src + offset)
        split.extend([entry["split"] == "train"] * len(c)); offset += len(c)
    if not clean:
        raise ValueError("Feature cache has no successful trajectories")
    return dict(clean=torch.cat(clean), corrupted=torch.cat(corrupted), source=torch.cat(source),
                is_train=torch.tensor(split, dtype=torch.bool), index=index,
                index_sha256=sha256(directory / "index.json"))
