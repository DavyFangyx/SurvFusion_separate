
from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def masked_mean(x, padding_mask=None):
    if padding_mask is None:
        return x.mean(dim=1)

    valid_mask = (~padding_mask).unsqueeze(-1).float()
    denom = valid_mask.sum(dim=1).clamp(min=1.0)
    return (x * valid_mask).sum(dim=1) / denom


def modality_dropout(available_mask, drop_prob=0.2, training=False):
    if not training or drop_prob <= 0:
        return available_mask

    keep_mask = available_mask.clone()
    random_drop = torch.rand_like(keep_mask.float()) < drop_prob
    keep_mask = keep_mask & (~random_drop)

    empty_rows = keep_mask.sum(dim=1) == 0
    if empty_rows.any():
        fallback_indices = torch.argmax(available_mask[empty_rows].float(), dim=1)
        keep_mask[empty_rows] = False
        keep_mask[empty_rows, fallback_indices] = True

    return keep_mask


class VAETransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        attn_out, _ = self.attn(
            self.attn_norm(x),
            self.attn_norm(x),
            self.attn_norm(x),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GeneClinicEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, nhead=8, mlp_dim=1024, num_layers=1, dropout=0.1):
        super().__init__()
        self.mu_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.logvar_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [VAETransformerBlock(input_dim, nhead=nhead, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(input_dim)
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        mu_query = self.mu_query.expand(batch_size, -1, -1)
        logvar_query = self.logvar_query.expand(batch_size, -1, -1)
        h = torch.cat([mu_query, logvar_query, x], dim=1)

        for block in self.blocks:
            h = block(h, padding_mask=None)

        h = self.norm(h)
        mu = self.to_mu(h[:, 0])
        logvar = self.to_logvar(h[:, 1]).clamp(min=-4.0, max=2.0)
        return mu, logvar


class VIBPPEG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, tokens, height, width, padding_mask=None):
        batch_size, _, dim = tokens.shape
        feat = tokens.transpose(1, 2).reshape(batch_size, dim, height, width)

        if padding_mask is not None:
            valid = (~padding_mask).reshape(batch_size, 1, height, width).float()
            feat = feat * valid
        else:
            valid = None

        feat = feat + self.proj7(feat) + self.proj5(feat) + self.proj3(feat)

        if valid is not None:
            feat = feat * valid

        return feat.flatten(2).transpose(1, 2)


class WSIEncoderVIB(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        token_dim=768,
        latent_dim=128,
        nhead=8,
        mlp_dim=1024,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.input_proj = nn.Linear(input_dim, token_dim)
        self.mu_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.logvar_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [VAETransformerBlock(token_dim, nhead=nhead, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.ppeg = VIBPPEG(token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.to_mu = nn.Linear(token_dim, latent_dim)
        self.to_logvar = nn.Linear(token_dim, latent_dim)

    def _pad_to_square(self, tokens, padding_mask=None):
        batch_size, num_tokens, dim = tokens.shape
        grid = int(np.ceil(np.sqrt(num_tokens)))
        target_len = grid * grid
        add_len = target_len - num_tokens

        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, num_tokens, device=tokens.device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.bool()

        if add_len > 0:
            pad_tokens = torch.zeros(batch_size, add_len, dim, device=tokens.device, dtype=tokens.dtype)
            pad_mask = torch.ones(batch_size, add_len, device=tokens.device, dtype=torch.bool)
            tokens = torch.cat([tokens, pad_tokens], dim=1)
            padding_mask = torch.cat([padding_mask, pad_mask], dim=1)

        return tokens, padding_mask, grid, grid

    def forward(self, x, padding_mask=None):
        batch_size = x.shape[0]
        tokens = self.input_proj(x)
        tokens, padding_mask, height, width = self._pad_to_square(tokens, padding_mask)

        mu_query = self.mu_query.expand(batch_size, -1, -1)
        logvar_query = self.logvar_query.expand(batch_size, -1, -1)
        query_mask = torch.zeros(batch_size, 2, device=tokens.device, dtype=torch.bool)

        h = torch.cat([mu_query, logvar_query, tokens], dim=1)
        full_padding_mask = torch.cat([query_mask, padding_mask], dim=1)

        h = self.blocks[0](h, padding_mask=full_padding_mask)
        feat_tokens = self.ppeg(h[:, 2:], height, width, padding_mask=padding_mask)
        h = torch.cat([h[:, :2], feat_tokens], dim=1)

        for block in self.blocks[1:]:
            h = block(h, padding_mask=full_padding_mask)

        h = self.norm(h)
        mu = self.to_mu(h[:, 0])
        logvar = self.to_logvar(h[:, 1]).clamp(min=-4.0, max=2.0)
        feat_tokens = h[:, 2:]
        return mu, logvar, feat_tokens, padding_mask


class WSITargetPoolingHead(nn.Module):
    def __init__(self, token_dim=768, output_dim=1024):
        super().__init__()
        self.proj = nn.Linear(token_dim, output_dim)

    def forward(self, tokens, padding_mask=None):
        pooled = masked_mean(tokens, padding_mask)
        return self.proj(pooled)


class GeneralizedPoE(nn.Module):
    def __init__(self, num_modalities=3):
        super().__init__()
        self.modality_logits = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, mus, logvars, available_mask):
        precision_terms = [torch.exp(-logvar.clamp(min=-4.0, max=2.0)) for logvar in logvars]
        stacked_mu = torch.stack(mus, dim=1)
        stacked_tau = torch.stack(precision_terms, dim=1)

        logits = self.modality_logits.unsqueeze(0).expand(available_mask.shape[0], -1)
        logits = logits.masked_fill(~available_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(available_mask, weights, torch.zeros_like(weights))

        weighted_tau = weights.unsqueeze(-1) * stacked_tau
        tau_joint = 1.0 + weighted_tau.sum(dim=1)
        mu_joint = (weighted_tau * stacked_mu).sum(dim=1) / tau_joint
        logvar_joint = torch.log(torch.reciprocal(tau_joint)).clamp(min=-4.0, max=2.0)
        return mu_joint, logvar_joint, weights


def reparameterize(mu, logvar, sample=True):
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class ReconstructionLoss(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.logvars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in dims
        })

    def forward(self, recon_dict, target_dict, available_mask):
        losses = {}
        total = 0.0
        for idx, (name, dim) in enumerate(self.dims.items()):
            sample_mask = available_mask[:, idx]
            if sample_mask.sum() == 0:
                losses[name] = recon_dict[name].sum() * 0.0
                continue

            recon = recon_dict[name][sample_mask]
            target = target_dict[name][sample_mask]
            mse = F.mse_loss(recon, target, reduction="none").sum(dim=1)
            logvar = self.logvars[name].clamp(min=-4.0, max=4.0)
            sigma2 = torch.exp(logvar)
            loss = mse / (2.0 * sigma2) + 0.5 * dim * logvar
            loss = loss.mean()
            losses[name] = loss
            total = total + loss

        losses["total"] = total
        return losses


class JeffreysDivergence(nn.Module):
    def forward(self, mu, logvar):
        sigma2 = torch.exp(logvar.clamp(min=-4.0, max=2.0))
        inv_sigma2 = torch.reciprocal(sigma2)
        value = 0.5 * (sigma2 + inv_sigma2 - 2.0 + mu.pow(2) * (1.0 + inv_sigma2))
        return value.sum(dim=1).mean()
