import torch
from realign_vla.losses import realign_loss, relative_error, token_cosine


def test_exact_final_formula_and_adapter_gradients():
    torch.manual_seed(3)
    corrupted, clean = torch.randn(3, 8, 12), torch.randn(3, 8, 12)
    residual = torch.randn(3, 8, 12, requires_grad=True)
    corrected = corrupted + residual
    token = 1 - torch.nn.functional.cosine_similarity(corrected, clean, dim=-1).mean(1).mean()
    rec = ((corrected-clean).flatten(1).norm(dim=1) / (clean.flatten(1).norm(dim=1)+1e-6)).mean()
    expected = token + rec + 1e-4 * residual.square().mean()
    loss, terms = realign_loss(corrupted, clean, residual)
    assert torch.equal(loss, expected)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert torch.isfinite(residual.grad).all()


def test_degenerate_targets_remain_finite():
    zeros = torch.zeros(2, 8, 12)
    assert torch.isfinite(relative_error(zeros, zeros)).all()
    assert torch.isfinite(token_cosine(zeros, zeros)).all()
    assert torch.isfinite(realign_loss(zeros, zeros, zeros)[0])
