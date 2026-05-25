import torch
import torch.nn as nn


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


class MLPGene(nn.Module):
    """
    Single-modal tabular gene model fed from RNA csv features.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 4,
        projection_dim: int = 512,
        dropout: float = 0.25,
    ):
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
        h = self.encoder(x)
        return self.classifier(h)

    def captum(self, omics):
        logits = self.forward(x_omic=omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return -torch.sum(survival, dim=1)


MLP_GENE = MLPGene
