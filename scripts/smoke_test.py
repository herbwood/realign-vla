"""CPU-only synthetic check; no VLA model, simulator, downloads or CUDA required."""
import argparse
import torch
from torch import nn
from realign_vla import TokenwiseLowRankAdapter, attach_realign_hook, remove_realign_hook
from realign_vla.intervention import assert_wrist_unchanged
from realign_vla.losses import realign_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    adapter = TokenwiseLowRankAdapter()
    features = torch.randn(1, 256, 2176)
    assert torch.equal(adapter(features), features)
    u, v = adapter.factors(features)
    assert u.shape == (1, 256, 16) and v.shape == (1, 2176, 16)
    loss, _ = realign_loss(features, features, adapter.residual(features))
    assert torch.isfinite(loss)
    loss.backward()
    assert adapter.u_head.weight.grad is not None
    class FrozenVLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = nn.Identity()
        def forward(self, features):
            return self.projector(features)
    with torch.no_grad():
        adapter.u_head.bias.fill_(0.1)
    vla = FrozenVLA()
    combined = torch.randn(1, 512, 2176, dtype=torch.bfloat16)
    hook = attach_realign_hook(vla, adapter)
    corrected = vla(combined)
    assert not torch.equal(corrected[:, :256], combined[:, :256])
    assert_wrist_unchanged(combined, corrected)
    remove_realign_hook(hook)
    assert torch.equal(vla(combined), combined)
    print("REALIGN_SYNTHETIC_SMOKE_PASS (CPU, original dimensions, wrist unchanged)")


if __name__ == "__main__":
    main()
