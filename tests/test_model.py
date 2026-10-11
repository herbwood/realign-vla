import torch
from torch import nn
from realign_vla.model import TokenwiseLowRankAdapter


def test_original_dimensions_factors_and_exact_identity():
    torch.manual_seed(0)
    model = TokenwiseLowRankAdapter()
    assert sum(p.numel() for p in model.parameters()) == 2279968
    assert not any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    features = torch.randn(2, 256, 2176)
    u, v = model.factors(features)
    assert u.shape == (2, 256, 16) and v.shape == (2, 2176, 16)
    assert torch.count_nonzero(u) == 0
    assert torch.equal(model(features), features)
    bf16 = features.bfloat16()
    assert model(bf16).dtype == bf16.dtype
    assert torch.equal(model(bf16), bf16)


def test_residual_factorization_and_rank_bound():
    torch.manual_seed(4)
    model = TokenwiseLowRankAdapter(feature_dim=12, num_tokens=8, hidden_dim=8, rank=3)
    nn.init.normal_(model.u_head.weight)
    features = torch.randn(2, 8, 12)
    u, v = model.factors(features)
    residual = model.residual(features)
    assert torch.equal(residual, u @ v.transpose(1, 2))
    assert int(torch.linalg.matrix_rank(residual[0], atol=1e-4)) <= 3
    assert torch.equal(model(features), features + residual)


def test_checkpoint_roundtrip(tmp_path):
    from realign_vla.utils import atomic_save, load_adapter
    model = TokenwiseLowRankAdapter(feature_dim=12, num_tokens=8, hidden_dim=8, rank=3)
    config = dict(feature_dim=12, num_tokens=8, hidden_dim=8, rank=3)
    atomic_save(dict(model_type="tokenwise_lowrank", config=config, state_dict=model.state_dict()), tmp_path / "adapter.pt")
    restored = load_adapter(tmp_path / "adapter.pt")
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad for p in restored.parameters())
