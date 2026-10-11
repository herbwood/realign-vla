"""Agent-view-only intervention on the frozen projector input."""
import torch
from torch import nn
from .model import TokenwiseLowRankAdapter


def assert_wrist_unchanged(before: torch.Tensor, after: torch.Tensor, agent_tokens: int = 256) -> None:
    if not torch.equal(before[:, agent_tokens:], after[:, agent_tokens:]):
        raise AssertionError("Wrist features changed")


def transform_features(
    features: torch.Tensor, adapter: TokenwiseLowRankAdapter | None, *,
    agent_tokens: int = 256, wrist_tokens: int = 256, feature_dim: int = 2176,
    identity: bool = False,
) -> torch.Tensor:
    if features.ndim != 3 or features.shape[1:] != (agent_tokens + wrist_tokens, feature_dim):
        raise ValueError(f"Expected [B,{agent_tokens+wrist_tokens},{feature_dim}], got {tuple(features.shape)}")
    if identity:
        return features
    if adapter is None:
        raise ValueError("An adapter is required unless identity=True")
    if next(adapter.parameters()).device != features.device:
        raise ValueError("Move the adapter to the visual features' device before attaching")
    corrected = adapter(features[:, :agent_tokens])
    return torch.cat([corrected, features[:, agent_tokens:]], dim=1)


class ReAlignHook:
    def __init__(self, vla: nn.Module, adapter: TokenwiseLowRankAdapter | None, *, identity: bool = False):
        if not hasattr(vla, "projector"):
            raise ValueError("VLA must expose its visual projector as vla.projector")
        self.vla, self.adapter, self.identity = vla, adapter, identity
        self._handle = None

    def _pre_hook(self, module, args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Projector must receive one combined visual feature tensor")
        with torch.no_grad():
            return (transform_features(args[0], self.adapter, identity=self.identity),)

    def attach(self):
        if self._handle is not None:
            raise RuntimeError("Hook already attached")
        self.vla.eval().requires_grad_(False)
        if self.adapter is not None:
            device = next(self.vla.parameters(), torch.empty(0)).device
            self.adapter.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self._handle = self.vla.projector.register_forward_pre_hook(self._pre_hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()


def attach_realign_hook(vla: nn.Module, adapter: TokenwiseLowRankAdapter | None = None, *, identity: bool = False) -> ReAlignHook:
    return ReAlignHook(vla, adapter, identity=identity).attach()


def remove_realign_hook(hook: ReAlignHook) -> None:
    hook.remove()
