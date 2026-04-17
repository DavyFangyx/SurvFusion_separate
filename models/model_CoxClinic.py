import torch
import torch.nn as nn


class CoxClinic(nn.Module):
    def __init__(self, input_dim, clinic_num_tokens=6):
        super().__init__()
        self.input_dim = input_dim
        self.clinic_num_tokens = clinic_num_tokens
        self.risk_head = nn.Linear(input_dim * clinic_num_tokens, 1, bias=False)

    def forward(self, **kwargs):
        clinic = kwargs["x_clinic"].float()
        if clinic.dim() == 2:
            clinic = clinic.unsqueeze(0)

        clinic = clinic.reshape(clinic.shape[0], -1)
        return self.risk_head(clinic)