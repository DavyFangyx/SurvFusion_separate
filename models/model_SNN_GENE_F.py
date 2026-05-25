import torch
import torch.nn as nn

from models.model_utils import SNN_Block


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


def init_max_weights(module):
    import math

    for child in module.modules():
        if isinstance(child, nn.Linear):
            stdv = 1.0 / math.sqrt(child.weight.size(1))
            child.weight.data.normal_(0, stdv)
            child.bias.data.zero_()


class SNNGeneFM(nn.Module):
    """
    Single-modal Gene Foundation SNN.
    Supports inputs shaped like [B, 3072] or [B, 4, 768].
    """

    def __init__(
        self,
        input_dim: int = 3072,
        model_size: str = "medium",
        n_classes: int = 4,
        use_input_ln: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        size_dict = {
            "medium": [512, 512, 256],
            "big": [1024, 1024, 1024, 256],
        }
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
        x = _extract_gene_fm_tensor(kwargs)
        x = self.input_norm(x)
        h = self.encoder(x)
        logits = self.classifier(h)
        if return_feats:
            return h, logits
        return logits


SNN_GENE_F = SNNGeneFM
