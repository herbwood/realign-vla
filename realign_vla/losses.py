"""Final token-cosine + normalized Frobenius + residual-square objective."""
import torch
from torch.nn import functional as F


def token_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame mean token cosine, using the source's default cosine epsilon."""
    return F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean(1)


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame ||prediction-target||_F / (||target||_F + 1e-6), not squared."""
    prediction, target = prediction.float(), target.float()
    return (prediction - target).flatten(1).norm(dim=1) / (target.flatten(1).norm(dim=1) + 1e-6)


def realign_loss(
    features: torch.Tensor, target: torch.Tensor, residual: torch.Tensor, *,
    lambda_rec: float = 1.0, lambda_res: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    corrected = features.float() + residual.float()
    terms = {
        "token": 1 - token_cosine(corrected, target).mean(),
        "reconstruction": relative_error(corrected, target).mean(),
        "residual": residual.float().square().mean(),
    }
    total = terms["token"] + lambda_rec * terms["reconstruction"] + lambda_res * terms["residual"]
    return total, terms
