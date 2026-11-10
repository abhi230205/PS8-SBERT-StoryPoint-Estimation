"""Training / inference routines for the SentenceCNN regressor."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .model import SentenceCNN
from .utils import mae, set_seed


def _to_target(y: torch.Tensor, transform: str) -> torch.Tensor:
    return torch.log1p(y) if transform == "log1p" else y


def _from_target(z: torch.Tensor, transform: str) -> torch.Tensor:
    return torch.expm1(z) if transform == "log1p" else z


def build_model(hp: dict, emb_dim: int) -> SentenceCNN:
    return SentenceCNN(
        emb_dim=emb_dim,
        channels=tuple(hp["channels"]),
        kernel_size=hp["kernel_size"],
        hidden=hp["hidden"],
        dropout=hp["dropout"],
    )


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, M: torch.Tensor, transform: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        z = model(X[i : i + batch_size].float(), M[i : i + batch_size])
        out.append(_from_target(z, transform).clamp(min=0.0))
    return torch.cat(out).cpu().numpy() if out else np.zeros(0)


def fit(
    X_tr: torch.Tensor,
    M_tr: torch.Tensor,
    y_tr: torch.Tensor,
    hp: dict,
    train_cfg: dict,
    seed: int,
    X_va: torch.Tensor | None = None,
    M_va: torch.Tensor | None = None,
    y_va: torch.Tensor | None = None,
    epochs: int | None = None,
):
    """Train a SentenceCNN.

    With a validation set, trains up to ``train_cfg['max_epochs']`` and early-stops on the
    validation MAE (patience ``train_cfg['patience']``), restoring the best weights.
    Without one, trains for exactly ``epochs`` epochs (used for the final refit on all
    development data).  Returns (model, info dict).
    """
    set_seed(seed)
    device = X_tr.device
    transform = train_cfg["target_transform"]
    model = build_model(hp, X_tr.shape[-1]).to(device)

    # Start the output bias at the median training target -> faster, more stable L1 training.
    with torch.no_grad():
        model.head[-1].bias.fill_(float(_to_target(y_tr, transform).median()))

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.L1Loss() if train_cfg["loss"] == "l1" else nn.SmoothL1Loss(beta=1.0)
    z_tr = _to_target(y_tr, transform)

    n = X_tr.shape[0]
    bs = train_cfg["batch_size"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    max_epochs = epochs if epochs is not None else train_cfg["max_epochs"]
    patience = train_cfg["patience"]

    best_state, best_val, best_epoch, wait = None, math.inf, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            if idx.numel() < 2 and n > 1:  # avoid degenerate final batch
                continue
            pred = model(X_tr[idx].float(), M_tr[idx])
            loss = loss_fn(pred, z_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step()
            tot += loss.item() * idx.numel()
        rec = {"epoch": epoch, "train_loss": tot / n}

        if X_va is not None:
            val_pred = predict(model, X_va, M_va, transform)
            val_mae = mae(y_va.cpu().numpy(), val_pred)
            rec["val_mae"] = val_mae
            if val_mae < best_val - 1e-6:
                best_val, best_epoch, wait = val_mae, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    history.append(rec)
                    break
        history.append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    return model, {"best_epoch": best_epoch, "best_val_mae": best_val if X_va is not None else None, "history": history}
