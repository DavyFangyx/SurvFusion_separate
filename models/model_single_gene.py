import torch
import torch.nn as nn

from models.model_utils import SNN_Block, init_max_weights


def _extract_omic_tensor(kwargs):
    x = kwargs.get("x_omic")
    if x is None:
        x = kwargs.get("data_omics")
    if x is None:
        raise KeyError("Expected `x_omic` or `data_omics` in kwargs.")
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    return x


def _extract_gene_fm_tensor(kwargs):
    x = kwargs.get("x_omic")
    if x is None:
        x = kwargs.get("data_omics")
    if x is None:
        raise KeyError("Expected `x_omic` or `data_omics` in kwargs.")
    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2 and x.shape[-1] != 3072:
        x = x.reshape(1, -1)
    elif x.dim() == 3:
        x = x.reshape(x.shape[0], -1)
    elif x.dim() > 3:
        raise ValueError(f"Unsupported Gene FM tensor shape: {tuple(x.shape)}")
    return x


class MLPGene(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 4, projection_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.projection_dim = projection_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim // 2, n_classes)

    def forward(self, **kwargs):
        x = _extract_omic_tensor(kwargs)
        return self.classifier(self.encoder(x))

    def captum(self, omics):
        logits = self.forward(x_omic=omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return -torch.sum(survival, dim=1)


class MLPGeneFM(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        n_classes: int = 4,
        model_size: str = "medium",
        dropout: float = 0.25,
        use_input_ln: bool = False,
    ):
        super().__init__()
        size_dict = {"medium": 512, "big": 1024}
        if model_size not in size_dict:
            raise ValueError(f"Unsupported model_size `{model_size}`.")
        projection_dim = size_dict[model_size]
        self.input_norm = nn.LayerNorm(input_dim) if use_input_ln else nn.Identity()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim // 2, n_classes)

    def forward(self, return_feats: bool = False, **kwargs):
        x = self.input_norm(_extract_gene_fm_tensor(kwargs))
        h = self.encoder(x)
        logits = self.classifier(h)
        return (h, logits) if return_feats else logits


class SNNGene(nn.Module):
    def __init__(
        self,
        input_dim: int = None,
        omic_input_dim: int = None,
        model_size_omic: str = "small",
        n_classes: int = 4,
    ):
        super().__init__()
        input_dim = omic_input_dim if omic_input_dim is not None else input_dim
        if input_dim is None:
            raise ValueError("Expected `input_dim` or `omic_input_dim`.")
        size_dict = {"small": [256, 256], "big": [1024, 1024, 1024, 256]}
        if model_size_omic not in size_dict:
            raise ValueError(f"Unsupported model_size_omic `{model_size_omic}`.")
        hidden = size_dict[model_size_omic]
        blocks = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for idx in range(len(hidden) - 1):
            blocks.append(SNN_Block(dim1=hidden[idx], dim2=hidden[idx + 1], dropout=0.25))
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, return_feats: bool = False, **kwargs):
        x = _extract_omic_tensor(kwargs)
        h = self.encoder(x)
        logits = self.classifier(h)
        return (h, logits) if return_feats else logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)


class SNNGeneFM(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        model_size: str = "medium",
        n_classes: int = 4,
        use_input_ln: bool = False,
    ):
        super().__init__()
        size_dict = {"medium": [512, 512, 256], "big": [1024, 1024, 1024, 256]}
        if model_size not in size_dict:
            raise ValueError(f"Unsupported model_size `{model_size}`.")
        hidden = size_dict[model_size]
        self.input_norm = nn.LayerNorm(input_dim) if use_input_ln else nn.Identity()
        blocks = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for idx in range(len(hidden) - 1):
            blocks.append(SNN_Block(dim1=hidden[idx], dim2=hidden[idx + 1], dropout=0.25))
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, return_feats: bool = False, **kwargs):
        x = self.input_norm(_extract_gene_fm_tensor(kwargs))
        h = self.encoder(x)
        logits = self.classifier(h)
        return (h, logits) if return_feats else logits


MLP_GENE = MLPGene
MLP_GENE_F = MLPGeneFM
SNN_GENE = SNNGene
SNN_GENE_F = SNNGeneFM


__all__ = [
    "MLPGene",
    "MLPGeneFM",
    "SNNGene",
    "SNNGeneFM",
    "MLP_GENE",
    "MLP_GENE_F",
    "SNN_GENE",
    "SNN_GENE_F",
]
