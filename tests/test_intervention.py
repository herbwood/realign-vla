import pytest
import torch
from torch import nn
from realign_vla import TokenwiseLowRankAdapter, attach_realign_hook, remove_realign_hook
from realign_vla.intervention import ReAlignHook, assert_wrist_unchanged, transform_features


class FakeVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Identity()
    def forward(self, features):
        return self.projector(features)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_only_agent_changes_and_wrist_is_bit_identical(dtype):
    vla, adapter = FakeVLA(), TokenwiseLowRankAdapter()
    with torch.no_grad():
        adapter.u_head.bias.fill_(0.2)
    combined = torch.randn(1, 512, 2176, dtype=dtype)
    original = combined.clone()
    with ReAlignHook(vla, adapter):
        corrected = vla(combined)
    assert corrected.dtype == dtype
    assert not torch.equal(corrected[:, :256], original[:, :256])
    assert_wrist_unchanged(original, corrected)
    assert torch.equal(combined, original)  # No in-place source modification.
    assert torch.equal(vla(combined), original)
    assert not any(p.requires_grad for p in adapter.parameters())


def test_noop_and_removal_after_exception():
    vla = FakeVLA()
    combined = torch.randn(1, 512, 2176)
    hook = attach_realign_hook(vla, identity=True)
    assert vla(combined) is combined
    remove_realign_hook(hook)
    with pytest.raises(RuntimeError):
        with ReAlignHook(vla, None, identity=True):
            raise RuntimeError("Simulated policy failure")
    assert not vla.projector._forward_pre_hooks


def test_wrong_view_layout_is_rejected():
    with pytest.raises(ValueError):
        transform_features(torch.zeros(1, 256, 2176), None, identity=True)
