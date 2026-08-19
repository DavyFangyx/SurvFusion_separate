from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[2]


@contextmanager
def _prepend_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == path_str:
            sys.path.pop(0)
        elif path_str in sys.path:
            sys.path.remove(path_str)


def _load_module(module_name: str, file_path: Path, extra_path: Path):
    with _prepend_path(extra_path):
        if str(ROOT_DIR) not in sys.path:
            sys.path.append(str(ROOT_DIR))
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def run_hgcn_from_args(args):
    hgcn_dir = ROOT_DIR / "models" / "missing_modality_baselines" / "third_party" / "HGCN" / "HGCN_code"
    module = _load_module("survpgc_hgcn_train", hgcn_dir / "train.py", hgcn_dir)
    cancer_type = args.study.replace("tcga_", "")
    data_pack_dir = getattr(args, "data_pack_dir", None)
    if data_pack_dir is None:
        data_pack_dir = ROOT_DIR / "models" / "missing_modality_baselines" / "third_party" / "HGCN" / "data_split" / cancer_type
    hgcn_args = SimpleNamespace(
        cancer_type=cancer_type,
        img_cox_loss_factor=5,
        rna_cox_loss_factor=1,
        cli_cox_loss_factor=5,
        train_use_type=['img', 'rna', 'cli'],
        format_of_coxloss='multi',
        add_mse_loss_of_mae=True,
        mse_loss_of_mae_factor=5,
        start_seed=getattr(args, "seed", 0),
        repeat_num=1,
        fusion_model='fusion_model_mae_2',
        drop_out_ratio=0.5,
        lr=getattr(args, "lr", 3e-5),
        epochs=getattr(args, "max_epochs", 60),
        batch_size=getattr(args, "batch_size", 32),
        n_hidden=512,
        out_classes=512,
        mix=True,
        if_adjust_lr=True,
        adjust_lr_ratio=0.5,
        if_fit_split=True,
        split_root=getattr(args, "split_root", str(ROOT_DIR / "splits" / "5foldcv")),
        split_dir=getattr(args, "split_dir", None),
        data_pack_dir=str(data_pack_dir),
        results_dir=getattr(args, "results_dir", "./results"),
        exp_group=getattr(args, "exp_group", "HGCN"),
        run_name=getattr(args, "run_name", f"{args.study}__hgcn"),
        details=getattr(args, "details", ""),
    )
    return module.main(hgcn_args)


def run_flex_moe_from_args(args):
    flex_dir = ROOT_DIR / "models" / "missing_modality_baselines" / "third_party" / "flex-moe"
    module = _load_module("survpgc_flex_moe_main", flex_dir / "main.py", flex_dir)
    flex_args = SimpleNamespace(
        device=getattr(args, "device", 0),
        data="survpgc",
        study=args.study,
        modality="WGC",
        preprocessed=True,
        initial_filling="mean",
        train_epochs=getattr(args, "max_epochs", 50),
        warm_up_epochs=getattr(args, "warmup_epochs", 5),
        batch_size=getattr(args, "batch_size", 8),
        lr=getattr(args, "lr", 1e-4),
        hidden_dim=128,
        top_k=4,
        num_patches=16,
        num_experts=16,
        num_routers=1,
        num_layers_enc=1,
        num_layers_fus=1,
        num_layers_pred=1,
        num_heads=4,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=getattr(args, "pin_memory", True),
        use_common_ids=False,
        dropout=0.5,
        gate_loss_weight=1e-2,
        split_csv=getattr(args, "split_csv", None) or (getattr(args, "split_dir", None) and str(Path(args.split_dir) / "splits_0.csv")),
        save=True,
        load_model=False,
        seed=getattr(args, "seed", 0),
        n_runs=1,
        results_dir=getattr(args, "results_dir", "./results"),
        exp_group=getattr(args, "exp_group", "FlexMoE"),
        run_name=getattr(args, "run_name", f"{args.study}__flex_moe"),
    )
    if not flex_args.split_csv:
        raise ValueError("`--split_csv` is required for `flex_moe` routing.")
    return module.main(flex_args)
