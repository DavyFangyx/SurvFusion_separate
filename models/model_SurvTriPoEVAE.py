import torch
import torch.nn as nn

from models.model_utils import (
    GeneralizedPoE,
    JeffreysDivergence,
    ModalityDecoder,
    ReconstructionLoss,
    TokenSetEncoder,
    WSIMILResampler,
    WSITargetPoolingHead,
    modality_dropout,
    reparameterize,
)


class SurvTriPoEVAE(nn.Module):
    def __init__(
        self,
        clinic_num_tokens,
        wsi_embedding_dim=1024,
        gene_embedding_dim=768,
        clinic_embedding_dim=512,
        gene_num_tokens=4,
        latent_dim=128,
        mmhid=256,
        label_dim=1,
        decoder_hidden_dim=512,
        poe_variant="A",
        poe_surv_lambda=1.0,
        modality_dropout_prob=0.2,
        transformer_dropout=0.1,
        transformer_layers=1,
        wsi_resampler_tokens=16,
        wsi_resampler_layers=2,
        selected_modalities="wsi,gene,clinic",
    ):
        super().__init__()
        self.gene_num_tokens = gene_num_tokens
        self.clinic_num_tokens = clinic_num_tokens
        self.wsi_embedding_dim = wsi_embedding_dim
        self.gene_embedding_dim = gene_embedding_dim
        self.clinic_embedding_dim = clinic_embedding_dim
        self.latent_dim = latent_dim
        self.mmhid = mmhid
        self.label_dim = label_dim
        self.poe_variant = poe_variant.upper()
        self.poe_surv_lambda = poe_surv_lambda
        self.modality_dropout_prob = modality_dropout_prob
        self.modality_names = ("wsi", "gene", "clinic")
        self.selected_modalities = tuple(selected_modalities.split(","))

        if self.poe_variant not in {"A", "B", "C"}:
            raise ValueError(f"Unsupported poe_variant `{poe_variant}`.")
        if not self.selected_modalities:
            raise ValueError("At least one modality must be selected.")
        if any(name not in self.modality_names for name in self.selected_modalities):
            raise ValueError(f"Unsupported selected_modalities `{selected_modalities}`.")

        self.training_stage = "stage1" if self.poe_variant in {"A", "B"} else "stage2"
        self.backbone_frozen = False
        self.selected_modality_mask = torch.tensor(
            [name in self.selected_modalities for name in self.modality_names],
            dtype=torch.bool,
        )

        self.wsi_resampler = WSIMILResampler(
            input_dim=wsi_embedding_dim,
            token_dim=768,
            num_tokens=wsi_resampler_tokens,
            nhead=8,
            mlp_dim=1024,
            num_layers=wsi_resampler_layers,
            dropout=transformer_dropout,
        )
        self.wsi_encoder = TokenSetEncoder(
            input_dim=768,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )
        self.gene_encoder = TokenSetEncoder(
            input_dim=gene_embedding_dim,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )
        self.clinic_encoder = TokenSetEncoder(
            input_dim=clinic_embedding_dim,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )

        self.wsi_target_head = WSITargetPoolingHead(token_dim=768, output_dim=768)
        self.poe = GeneralizedPoE(num_modalities=3)

        self.decoder_wsi = ModalityDecoder(latent_dim, decoder_hidden_dim, 768)
        self.decoder_gene = ModalityDecoder(latent_dim, decoder_hidden_dim, gene_num_tokens * gene_embedding_dim)
        self.decoder_clinic = ModalityDecoder(latent_dim, decoder_hidden_dim, clinic_num_tokens * clinic_embedding_dim)

        self.reconstruction_loss = ReconstructionLoss({
            "wsi": 768,
            "gene": gene_num_tokens * gene_embedding_dim,
            "clinic": clinic_num_tokens * clinic_embedding_dim,
        })
        self.jeffreys = JeffreysDivergence()

        self.fuse_fc = nn.Sequential(
            nn.Linear(latent_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(mmhid, label_dim)
        self.linear_probe = nn.Linear(latent_dim, label_dim)

        self._cached_outputs = None

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            frozen_modules = [
                self.wsi_resampler,
                self.wsi_encoder,
                self.gene_encoder,
                self.clinic_encoder,
                self.wsi_target_head,
                self.poe,
                self.decoder_wsi,
                self.decoder_gene,
                self.decoder_clinic,
                self.reconstruction_loss,
            ]
            for module in frozen_modules:
                module.eval()
            self.fuse_fc.eval()
            self.classifier.eval()
            self.linear_probe.train(mode)
        return self

    def set_training_stage(self, stage):
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported training stage `{stage}`.")
        self.training_stage = stage

    def freeze_backbone_for_probe(self):
        modules = [
            self.wsi_resampler,
            self.wsi_encoder,
            self.gene_encoder,
            self.clinic_encoder,
            self.wsi_target_head,
            self.poe,
            self.decoder_wsi,
            self.decoder_gene,
            self.decoder_clinic,
            self.reconstruction_loss,
        ]
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        for parameter in self.linear_probe.parameters():
            parameter.requires_grad = True

        for parameter in self.fuse_fc.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

        self.backbone_frozen = True
        self.training_stage = "stage2"

    def get_cached_outputs(self):
        if self._cached_outputs is None:
            raise RuntimeError("No cached outputs available. Run forward first.")
        return self._cached_outputs

    def get_vae_loss(self, beta=1.0):
        cached = self.get_cached_outputs()
        return cached["recon_total"] + beta * cached["jeffreys"]

    def combine_loss(self, survival_loss, beta=1.0):
        if self.poe_variant == "C":
            return self.get_vae_loss(beta=beta) + self.poe_surv_lambda * survival_loss
        return survival_loss

    def _reshape_gene(self, x_omic):
        if x_omic.dim() == 2:
            return x_omic.reshape(x_omic.shape[0], self.gene_num_tokens, self.gene_embedding_dim)
        return x_omic

    def _reshape_clinic(self, x_clinic):
        if x_clinic.dim() == 2:
            return x_clinic.reshape(x_clinic.shape[0], self.clinic_num_tokens, self.clinic_embedding_dim)
        return x_clinic

    def _build_available_mask(self, batch_size, device):
        available_mask = self.selected_modality_mask.to(device=device).unsqueeze(0).expand(batch_size, -1).clone()
        use_dropout = self.training and len(self.selected_modalities) > 1 and (
            self.training_stage == "stage1" or self.poe_variant == "C"
        )
        return modality_dropout(available_mask, self.modality_dropout_prob, training=use_dropout)

    def _select_latent_for_survival(self, mu_joint, z_joint):
        if self.training_stage == "stage1":
            return z_joint
        if self.poe_variant == "A":
            return mu_joint
        if self.training:
            return z_joint
        return mu_joint

    def forward(self, x_path, x_omic, x_clinic, wsi_mask=None):
        x_omic = self._reshape_gene(x_omic.float())
        x_clinic = self._reshape_clinic(x_clinic.float())
        x_path = x_path.float()

        batch_size = x_path.shape[0]
        available_mask = self._build_available_mask(batch_size, x_path.device)

        wsi_tokens = self.wsi_resampler(x_path, padding_mask=wsi_mask)
        mu_wsi, logvar_wsi = self.wsi_encoder(wsi_tokens)
        mu_gene, logvar_gene = self.gene_encoder(x_omic)
        mu_clinic, logvar_clinic = self.clinic_encoder(x_clinic)

        mu_joint, logvar_joint, poe_weights = self.poe(
            mus=[mu_wsi, mu_gene, mu_clinic],
            logvars=[logvar_wsi, logvar_gene, logvar_clinic],
            available_mask=available_mask,
        )

        should_sample = self.training and self.poe_variant in {"B", "C"}
        if self.training_stage == "stage1":
            should_sample = True
        z_joint = reparameterize(mu_joint, logvar_joint, sample=should_sample)

        recon_wsi = self.decoder_wsi(z_joint)
        recon_gene = self.decoder_gene(z_joint)
        recon_clinic = self.decoder_clinic(z_joint)

        target_wsi = self.wsi_target_head(wsi_tokens)
        target_gene = x_omic.reshape(batch_size, -1)
        target_clinic = x_clinic.reshape(batch_size, -1)

        recon_losses = self.reconstruction_loss(
            recon_dict={
                "wsi": recon_wsi,
                "gene": recon_gene,
                "clinic": recon_clinic,
            },
            target_dict={
                "wsi": target_wsi,
                "gene": target_gene,
                "clinic": target_clinic,
            },
            available_mask=available_mask,
        )
        jeffreys = self.jeffreys(mu_joint, logvar_joint)
        survival_latent = self._select_latent_for_survival(mu_joint, z_joint)
        if self.poe_variant == "A":
            risk = self.linear_probe(survival_latent)
            fused = survival_latent
        else:
            fused = self.fuse_fc(survival_latent)
            risk = self.classifier(fused)

        self._cached_outputs = {
            "risk": risk,
            "fused": fused,
            "z_joint": z_joint,
            "mu_joint": mu_joint,
            "logvar_joint": logvar_joint,
            "poe_weights": poe_weights,
            "available_mask": available_mask,
            "wsi_tokens": wsi_tokens,
            "recon_losses": recon_losses,
            "recon_total": recon_losses["total"],
            "jeffreys": jeffreys,
        }
        return risk
