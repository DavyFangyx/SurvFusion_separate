import torch
import torch.nn as nn

from models.model_utils import SNN_Block, init_max_weights


def _extract_clinic_tensor(kwargs, expect_tokens=None, expect_dim=None):
    x = kwargs.get("x_clinic")
    if x is None:
        x = kwargs.get("data_clinic")
    if x is None:
        raise KeyError("Expected `x_clinic` or `data_clinic` in kwargs.")
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0)
    elif x.dim() != 3:
        raise ValueError(f"Unsupported clinic tensor shape: {tuple(x.shape)}")
    if expect_tokens is not None and x.shape[1] != expect_tokens:
        raise ValueError(f"clinic token mismatch: got {x.shape[1]}, expect {expect_tokens}")
    if expect_dim is not None and x.shape[2] != expect_dim:
        raise ValueError(f"clinic feat_dim mismatch: got {x.shape[2]}, expect {expect_dim}")
    return x


class MLPClinic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 4,
        projection_dim: int = 512,
        dropout: float = 0.1,
        clinic_num_tokens: int = 6,
        pooling: str = "flatten",
    ):
        super().__init__()
        if pooling not in {"mean", "flatten"}:
            raise ValueError(f"Unsupported pooling `{pooling}`.")
        hidden_dim = projection_dim // 4
        self.input_dim = input_dim
        self.pooling = pooling
        self.clinic_num_tokens = clinic_num_tokens
        self.token_net = nn.Sequential(
            nn.Linear(input_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        classifier_in_dim = hidden_dim if pooling == "mean" else hidden_dim * clinic_num_tokens
        self.classifier = nn.Linear(classifier_in_dim, n_classes)

    def forward(self, return_feats: bool = False, **kwargs):
        x = _extract_clinic_tensor(
            kwargs,
            expect_tokens=self.clinic_num_tokens,
            expect_dim=self.input_dim,
        )
        h = self.token_net(x)
        pooled = h.mean(dim=1) if self.pooling == "mean" else h.reshape(h.shape[0], -1)
        logits = self.classifier(pooled)
        return (pooled, logits) if return_feats else logits

    def captum(self, clinic):
        logits = self.forward(x_clinic=clinic)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return -torch.sum(survival, dim=1)


class SNNClinic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_size_omic: str = "small",
        n_classes: int = 4,
        clinic_num_tokens: int = 6,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "flatten"}:
            raise ValueError(f"Unsupported pooling `{pooling}`.")
        size_dict = {"small": [256, 256], "big": [1024, 1024, 1024, 256]}
        if model_size_omic not in size_dict:
            raise ValueError(f"Unsupported model_size_omic `{model_size_omic}`.")
        hidden = size_dict[model_size_omic]
        self.input_dim = input_dim
        blocks = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for idx in range(len(hidden) - 1):
            blocks.append(SNN_Block(dim1=hidden[idx], dim2=hidden[idx + 1], dropout=0.1))
        self.pooling = pooling
        self.clinic_num_tokens = clinic_num_tokens
        self.encoder = nn.Sequential(*blocks)
        classifier_in_dim = hidden[-1] if pooling == "mean" else hidden[-1] * clinic_num_tokens
        self.classifier = nn.Linear(classifier_in_dim, n_classes)
        init_max_weights(self)

    def forward(self, return_feats: bool = False, **kwargs):
        x = _extract_clinic_tensor(
            kwargs,
            expect_tokens=self.clinic_num_tokens,
            expect_dim=self.input_dim,
        )
        h = self.encoder(x)
        pooled = h.mean(dim=1) if self.pooling == "mean" else h.reshape(h.shape[0], -1)
        logits = self.classifier(pooled)
        return (pooled, logits) if return_feats else logits


class CoxClinic(nn.Module):
    def __init__(self, input_dim, clinic_num_tokens=6):
        super().__init__()
        self.input_dim = input_dim
        self.clinic_num_tokens = clinic_num_tokens
        self.risk_head = nn.Linear(input_dim * clinic_num_tokens, 1, bias=False)

    def forward(self, **kwargs):
        x = _extract_clinic_tensor(
            kwargs,
            expect_tokens=self.clinic_num_tokens,
            expect_dim=self.input_dim,
        )
        return self.risk_head(x.reshape(x.shape[0], -1))


MLP_CLINIC = MLPClinic
SNN_CLINIC = SNNClinic


__all__ = [
    "MLPClinic",
    "SNNClinic",
    "CoxClinic",
    "MLP_CLINIC",
    "SNN_CLINIC",
]
