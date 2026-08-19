"""Model B ablation: skip stage1 pretraining and start stage2 from random init."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE


class SurvTriPoEVAE_BNoPretrain(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "B"
        super().__init__(*args, **kwargs)
        self.training_stage = "stage2"
        self.backbone_frozen = False
        self.skip_stage1 = True

    def set_training_stage(self, stage):
        if stage == "stage1":
            raise ValueError("SurvTriPoEVAE_BNoPretrain skips stage1 by design.")
        return super().set_training_stage("stage2")
