import sys
from importlib import import_module
from types import ModuleType

_gene = import_module(__name__ + ".model_single_gene")
_clinic = import_module(__name__ + ".model_single_clinic")
_wsi = import_module(__name__ + ".model_single_wsi")

from models.model_single_clinic import CoxClinic, MLPClinic, MLP_CLINIC, SNNClinic, SNN_CLINIC
from models.model_single_gene import MLPGene, MLPGeneFM, MLP_GENE, MLP_GENE_F, SNNGene, SNNGeneFM, SNN_GENE, SNN_GENE_F
from models.model_single_wsi import MLPWSI, MLP_WSI

_legacy_single_modal = ModuleType(__name__ + ".model_single_modal")
for _module in (_gene, _clinic, _wsi):
    for _name in getattr(_module, "__all__", []):
        setattr(_legacy_single_modal, _name, getattr(_module, _name))
sys.modules.setdefault(__name__ + ".model_single_modal", _legacy_single_modal)
sys.modules.setdefault(__name__ + ".model_G", _gene)
sys.modules.setdefault(__name__ + ".model_C", _clinic)
sys.modules.setdefault(__name__ + ".model_P", _wsi)
sys.modules.setdefault(__name__ + ".model_MLP_GENE", _gene)
sys.modules.setdefault(__name__ + ".model_MLP_GENE_F", _gene)
sys.modules.setdefault(__name__ + ".model_SNN_GENE", _gene)
sys.modules.setdefault(__name__ + ".model_SNN_GENE_F", _gene)
sys.modules.setdefault(__name__ + ".model_MLP_CLINIC", _clinic)
sys.modules.setdefault(__name__ + ".model_SNN_CLINIC", _clinic)
sys.modules.setdefault(__name__ + ".model_CoxClinic", _clinic)
sys.modules.setdefault(__name__ + ".model_MLPWSI", _wsi)
sys.modules.setdefault(__name__ + ".model_MLP_WSI", _wsi)

__all__ = [
    "CoxClinic",
    "MLPClinic",
    "MLPWSI",
    "MLP_WSI",
    "MLPGene",
    "MLPGeneFM",
    "MLP_CLINIC",
    "MLP_GENE",
    "MLP_GENE_F",
    "SNNClinic",
    "SNNGene",
    "SNNGeneFM",
    "SNN_CLINIC",
    "SNN_GENE",
    "SNN_GENE_F",
]
