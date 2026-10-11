"""Small configuration, checkpoint and reproducibility helpers."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from .model import TokenwiseLowRankAdapter


def load_config(path: str | Path) -> dict:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    return config


def seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_kwargs(config: dict) -> dict:
    return dict(feature_dim=config.get("feature_dim", config.get("dim", 2176)),
                num_tokens=config.get("agent_tokens", config.get("num_tokens", 256)),
                hidden_dim=config.get("hidden_dim", config.get("hidden", 256)),
                rank=config.get("rank", 16), layers=config.get("layers", 2),
                heads=config.get("heads", 4), alpha=config.get("alpha", 1.0))


def load_adapter(path: str | Path, device: str | torch.device = "cpu") -> TokenwiseLowRankAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") not in ("tokenwise_lowrank", "lowrank_nomix"):
        raise ValueError("Expected the final tokenwise NoMixer adapter checkpoint")
    model = TokenwiseLowRankAdapter(**model_kwargs(checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def atomic_save(checkpoint: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def warmup_cosine(step: int, *, warmup_steps: int, total_steps: int) -> float:
    import math
    return min(1, (step + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
