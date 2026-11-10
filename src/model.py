"""Three-layer 1-D CNN regressor over a sequence of SBERT sentence embeddings.

Each issue is represented as a matrix X in R^{L x d}: row 0 is the SBERT embedding of the
issue title, rows 1..L-1 are the SBERT embeddings of the description sentences (in reading
order), zero-padded to a fixed length L with a boolean mask.  The CNN slides along the
sentence axis, so each filter looks at windows of ``kernel_size`` consecutive sentences.

    (B, L, d) -> transpose -> (B, d, L)
    Conv1d(d  -> c1, k) + ReLU + Dropout
    Conv1d(c1 -> c2, k) + ReLU + Dropout
    Conv1d(c2 -> c3, k) + ReLU
    masked global max-pool  ||  masked global mean-pool      -> (B, 2*c3)
    Linear(2*c3 -> hidden) + ReLU + Dropout -> Linear(hidden -> 1)   (story points)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SentenceCNN(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        channels: tuple[int, int, int] = (256, 128, 64),
        kernel_size: int = 3,
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(emb_dim, c1, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size, padding=pad)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size, padding=pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * c3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) float, mask: (B, L) bool (True = real sentence)
        m = mask.unsqueeze(1).to(x.dtype)  # (B, 1, L)
        h = x.transpose(1, 2) * m  # (B, d, L)
        h = self.drop(self.act(self.conv1(h))) * m
        h = self.drop(self.act(self.conv2(h))) * m
        h = self.act(self.conv3(h))
        neg_inf = torch.finfo(h.dtype).min
        h_max = h.masked_fill(m == 0, neg_inf).max(dim=2).values
        h_mean = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.head(torch.cat([h_max, h_mean], dim=1)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
