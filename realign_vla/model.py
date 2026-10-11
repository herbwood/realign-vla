"""The final rank-16 NoMixer adapter, with original checkpoint state keys."""
import torch
from torch import nn


class PerTokenMLPBlock(nn.Module):
    def __init__(self, hidden: int = 256, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mult * hidden), nn.GELU(), nn.Linear(mult * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class TokenwiseLowRankAdapter(nn.Module):
    def __init__(
        self, feature_dim: int = 2176, num_tokens: int = 256, hidden_dim: int = 256,
        rank: int = 16, layers: int = 2, heads: int = 4, alpha: float = 1.0,
    ):
        super().__init__()
        self.dim, self.num_tokens, self.rank, self.alpha = feature_dim, num_tokens, rank, alpha
        self.in_norm = nn.LayerNorm(feature_dim)
        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        # The checkpoint-producing constructor initialized a base attention block
        # before replacing its mixer. Preserve those RNG draws for seeded parity.
        # This temporary object is discarded; the final model has no attention mixer.
        nn.TransformerEncoderLayer(
            hidden_dim, heads, dim_feedforward=4 * hidden_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.u_head = nn.Linear(hidden_dim, rank)
        nn.init.zeros_(self.u_head.weight)
        nn.init.zeros_(self.u_head.bias)
        self.slot_logits = nn.Linear(hidden_dim, rank)
        self.v_out = nn.Linear(hidden_dim, feature_dim, bias=False)
        self.v0 = nn.Parameter(torch.randn(rank, feature_dim) * 0.02)
        self.mixer = nn.Sequential(*[PerTokenMLPBlock(hidden_dim) for _ in range(layers)])

    def factors(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B,N,D] -> U:[B,N,r], V:[B,D,r]; adapter arithmetic is FP32."""
        if features.ndim != 3 or features.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(f"Expected [B,{self.num_tokens},{self.dim}], got {tuple(features.shape)}")
        h = nn.functional.gelu(self.in_proj(self.in_norm(features.float()))) + self.pos
        h = self.out_norm(self.mixer(h))
        coefficients = self.u_head(h)  # Token-specific coefficients U (also denoted C).
        attention = torch.softmax(self.slot_logits(h), dim=1)  # Pool over tokens, one slot/rank.
        slots = attention.transpose(1, 2) @ h  # [B,r,hidden_dim]
        directions = (self.v0 + self.v_out(slots)).transpose(1, 2)  # [B,D,r]
        return coefficients, directions

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        coefficients, directions = self.factors(features)
        return coefficients @ directions.transpose(1, 2)  # [B,N,D], rank <= r.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features.float() + self.alpha * self.residual(features)).to(features.dtype)
