import torch
import torch.nn as nn


def _extract_gene_fm_tensor(kwargs):
    x = kwargs.get("x_omic")
    if x is None:
        x = kwargs.get("data_omics")
    if x is None:
        raise KeyError("Expected `x_omic` or `data_omics` in kwargs.")

    x = x.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        if x.shape[-1] != 3072:
            x = x.reshape(1, -1)
    elif x.dim() == 3:
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unsupported Gene FM tensor shape: {tuple(x.shape)}")
    return x


class MLPGeneFM(nn.Module):
    """
    Single-modal Gene Foundation model.
    Supports inputs shaped like [B, 3072] or [B, 4, 768].
    """

    def __init__(
        self,
        input_dim: int = 3072,
        n_classes: int = 4,
        model_size: str = "medium",
        dropout: float = 0.25,
        use_input_ln: bool = False,
    ):
        super().__init__()
        size_dict = {
            "medium": 512,
            "big": 1024,
        }
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

    def forward(self, **kwargs):
        x = _extract_gene_fm_tensor(kwargs)
        x = self.input_norm(x)
        h = self.encoder(x)
        return self.classifier(h)


MLP_GENE_F = MLPGeneFM
