import torch
import torch.nn as nn

from models.model_utils import SNN_Block


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


def init_max_weights(module):
    import math

    for child in module.modules():
        if isinstance(child, nn.Linear):
            stdv = 1.0 / math.sqrt(child.weight.size(1))
            child.weight.data.normal_(0, stdv)
            child.bias.data.zero_()


class SNNGene(nn.Module):
    """
    Single-modal tabular gene SNN fed from RNA csv features.
    """

    def __init__(
        self,
        input_dim: int = None,
        omic_input_dim: int = None,
        model_size_omic: str = "small",
        n_classes: int = 4,
    ):
        super().__init__()
        self.n_classes = n_classes
        input_dim = omic_input_dim if omic_input_dim is not None else input_dim
        if input_dim is None:
            raise ValueError("Expected `input_dim` or `omic_input_dim`.")
        size_dict = {
            "small": [256, 256],
            "big": [1024, 1024, 1024, 256],
        }
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
        if return_feats:
            return h, logits
        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)


SNN_GENE = SNNGene
