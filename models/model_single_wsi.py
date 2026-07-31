import torch
import torch.nn as nn
from torch.nn import ReLU

from models.model_utils import init_max_weights
from models.layers.cross_attention import FeedForward


class MLPWSI(nn.Module):
    def __init__(self, wsi_embedding_dim=1024, dropout=0.1, num_classes=4, wsi_projection_dim=256, device="cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
            ReLU(),
        )
        self.feed_forward = FeedForward(self.wsi_projection_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim)
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes),
        )
        self.device = device
        init_max_weights(self)

    def forward(self, **kwargs):
        wsi = kwargs.get("data_WSI")
        if wsi is None:
            wsi = kwargs.get("x_path")
        if wsi is None:
            raise KeyError("Expected `data_WSI` or `x_path` in kwargs.")
        wsi_embed = self.wsi_projection_net(wsi)
        embedding = torch.mean(wsi_embed, dim=1)
        return self.to_logits(embedding)


MLP_WSI = MLPWSI


__all__ = ["MLPWSI", "MLP_WSI"]
