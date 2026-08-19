from __future__ import annotations

from itertools import combinations
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import masked_mean
from utils.loss_func import CoxSurvLoss


MODALITY_NAMES = ("wsi", "gene", "clinic")


def as_bool_tensor(value, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.bool).view(-1)
    return torch.as_tensor(value, device=device, dtype=torch.bool).view(-1)


def normalize_avail(avail, *, device: torch.device) -> torch.Tensor:
    if avail is None:
        raise ValueError("Expected `avail` to be provided by the dataloader.")
    masks = []
    for name in MODALITY_NAMES:
        if name not in avail:
            raise KeyError(f"Missing availability key `{name}`.")
        masks.append(as_bool_tensor(avail[name], device=device))
    return torch.stack(masks, dim=1)


def safe_flatten(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x.reshape(x.shape[0], -1)


def maybe_tokenize_tokens(x: torch.Tensor, token_dim: int | None = None) -> torch.Tensor:
    if x is None:
        raise ValueError("Missing modality tensor.")
    x = x.float()
    if x.dim() == 2:
        if token_dim is not None and x.shape[1] % token_dim == 0:
            return x.reshape(x.shape[0], -1, token_dim)
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")


def tokenize_wsi(x_path: torch.Tensor, *, wsi_mask=None, resampler: WSIMILResampler | None = None) -> torch.Tensor:
    x_path = x_path.float()
    if x_path.dim() == 2:
        x_path = x_path.unsqueeze(0)
    if resampler is None:
        return masked_mean(x_path, wsi_mask).unsqueeze(1)
    return resampler(x_path, padding_mask=wsi_mask)


def cox_loss(risk: torch.Tensor, event_time: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
    return CoxSurvLoss(reduction="mean")(risk, event_time, censor)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def prior_poe(mu_list: list[torch.Tensor], logvar_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prior_mu = torch.zeros_like(mu_list[0])
    prior_logvar = torch.zeros_like(logvar_list[0])
    mus = torch.stack([*mu_list, prior_mu], dim=0)
    logvars = torch.stack([*logvar_list, prior_logvar], dim=0)
    return stable_poe(mus, logvars)


def stable_poe(mus: torch.Tensor, logvars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(mus) == 1:
        return mus[0], logvars[0]
    ln_inv_vars = torch.stack([-l for l in logvars])
    ln_var = -torch.logsumexp(ln_inv_vars, dim=0)
    joint_mu = (torch.exp(ln_inv_vars) * mus).sum(dim=0) * torch.exp(ln_var)
    return joint_mu, ln_var


def all_nonempty_subsets(items: Iterable[str]) -> list[tuple[str, ...]]:
    names = list(items)
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(names) + 1):
        subsets.extend(combinations(names, r))
    return subsets


def moment_match_gaussian(mus: list[torch.Tensor], logvars: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked_mu = torch.stack(mus, dim=0)
    stacked_logvar = torch.stack(logvars, dim=0)
    weights = torch.full((stacked_mu.shape[0], 1, 1), 1.0 / stacked_mu.shape[0], device=stacked_mu.device, dtype=stacked_mu.dtype)
    mean = (weights * stacked_mu).sum(dim=0)
    second_moment = (weights * (torch.exp(stacked_logvar) + stacked_mu.pow(2))).sum(dim=0)
    var = (second_moment - mean.pow(2)).clamp(min=1e-6)
    return mean, var.log().clamp(min=-4.0, max=2.0)


class CoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
