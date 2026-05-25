import torch
import torch.nn as nn

from models.model_utils import SNN_Block


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


def init_max_weights(module):
    import math

    for child in module.modules():
        if isinstance(child, nn.Linear):
            stdv = 1.0 / math.sqrt(child.weight.size(1))
            child.weight.data.normal_(0, stdv)
            child.bias.data.zero_()


class SNNClinic(nn.Module):
    """
    Single-modal clinic SNN with configurable token aggregation.
    """

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

        size_dict = {
            "small": [256, 256],
            "big": [1024, 1024, 1024, 256],
        }
        if model_size_omic not in size_dict:
            raise ValueError(f"Unsupported model_size_omic `{model_size_omic}`.")

        hidden = size_dict[model_size_omic]
        blocks = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for idx in range(len(hidden) - 1):
            blocks.append(SNN_Block(dim1=hidden[idx], dim2=hidden[idx + 1], dropout=0.1))

        self.pooling = pooling
        self.clinic_num_tokens = clinic_num_tokens
        self.encoder = nn.Sequential(*blocks)
        classifier_in_dim = hidden[-1] if pooling == "mean" else hidden[-1] * clinic_num_tokens
        self.classifier = nn.Linear(classifier_in_dim, n_classes)
        self.n_classes = n_classes
        init_max_weights(self)

    def forward(self, return_feats: bool = False, **kwargs):
        x = _extract_clinic_tensor(kwargs)
        h = self.encoder(x)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h.reshape(h.shape[0], -1)

        logits = self.classifier(pooled)
        if return_feats:
            return pooled, logits
        return logits


SNN_CLINIC = SNNClinic
