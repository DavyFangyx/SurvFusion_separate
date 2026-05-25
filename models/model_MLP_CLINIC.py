import torch
import torch.nn as nn


def _extract_clinic_tensor(kwargs):
    x = kwargs.get("x_clinic")
    if x is None:
        x = kwargs.get("data_clinic")
    if x is None:
        raise KeyError("Expected `x_clinic` or `data_clinic` in kwargs.")

    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    elif x.dim() != 3:
        raise ValueError(f"Unsupported clinic tensor shape: {tuple(x.shape)}")
    return x


class MLPClinic(nn.Module):
    """
    Single-modal clinic MLP with configurable token aggregation.
    """

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

        self.pooling = pooling
        self.clinic_num_tokens = clinic_num_tokens
        self.projection_dim = projection_dim
        hidden_dim = projection_dim // 4

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

    def forward(self, **kwargs):
        x = _extract_clinic_tensor(kwargs)
        h = self.token_net(x)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h.reshape(h.shape[0], -1)

        return self.classifier(pooled)

    def captum(self, clinic):
        logits = self.forward(x_clinic=clinic)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return -torch.sum(survival, dim=1)


MLP_CLINIC = MLPClinic
