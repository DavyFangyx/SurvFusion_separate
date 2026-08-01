import os
from copy import deepcopy
from timeit import default_timer as timer
from warnings import simplefilter

import pandas as pd

from datasets.dataset_survival import SurvivalDatasetFactory
from main import _init_wandb_run, _write_filter_log
from utils.core_utils import _train_val_test
from utils.general_utils import _prepare_for_experiment
from utils.optuna_utils import (
    build_optuna_components,
    build_trial_args,
    ensure_optuna_available,
    resolve_optuna_storage,
    sample_survtri_poe_vae_model_c,
    save_study_artifacts,
)
from utils.process_args import _process_args

simplefilter(action="ignore", category=FutureWarning)


def _build_dataset_factory(args):
    return SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        data_dir=args.data_root_dir,
        clinical_file=args.clinical_file,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat=True if args.modality == "mcat" else False,
        is_survpath=True if args.modality == "survpath" else False,
        is_survpgc=True if args.modality == "survpgc" else False,
        is_survpgc_f=True if args.modality in (
            "survpgc_f",
            "survfusion_separate",
            "survfusion_noalign",
            "survfusion_joint",
            "survtri_snn_concat",
            "survtri_snn_mhsa",
            "survtri_mlp_concat",
            "survtri_mlp_mhsa",
            "survtri_poe_vae",
        ) else False,
        is_survpc=True if args.modality == "survpc" else False,
        is_survpc_f=True if args.modality == "survpc_f" else False,
        type_of_pathway=args.type_of_path,
    )


def _objective_factory(base_args):
    import optuna

    def objective(trial):
        if base_args.modality != "survtri_poe_vae" or base_args.poe_variant != "C":
            raise ValueError("The minimal Optuna tuner currently supports only survtri_poe_vae with poe_variant=C.")

        sampled_params = sample_survtri_poe_vae_model_c(trial, base_args)
        trial_args = build_trial_args(base_args, trial, sampled_params, optuna.TrialPruned)
        trial_args = _prepare_for_experiment(trial_args)
        trial_args.dataset_factory = _build_dataset_factory(trial_args)
        _write_filter_log(trial_args)

        fold = int(trial_args.optuna_fold)
        trial_args.wandb_run = _init_wandb_run(trial_args, fold)

        try:
            datasets = trial_args.dataset_factory.return_splits(
                trial_args,
                csv_path=f"{trial_args.split_dir}/splits_{fold}.csv",
                fold=fold,
            )
            _train_val_test(datasets, fold, trial_args)
            val_result_path = os.path.join(trial_args.results_dir, f"val_result_fold{fold}.csv")
            val_df = pd.read_csv(val_result_path)
            best_val_cindex = float(val_df["val_cindex"].iloc[0])
            if trial_args.wandb_run is not None:
                trial_args.wandb_run.summary["optuna/best_val_cindex"] = best_val_cindex
                trial_args.wandb_run.summary["optuna/trial_number"] = trial.number
            return best_val_cindex
        finally:
            if getattr(trial_args, "wandb_run", None) is not None:
                import wandb
                wandb.finish()
                trial_args.wandb_run = None

    return objective


def main(args):
    ensure_optuna_available()
    import optuna

    if args.modality != "survtri_poe_vae":
        raise ValueError("main_tune_optuna.py currently supports only `--modality survtri_poe_vae`.")
    if args.poe_variant != "C":
        raise ValueError("The minimal Optuna version currently supports only `--poe_variant C`.")

    study_name = args.optuna_study_name or f"{args.study}_{args.modality}_{args.poe_variant}_fold{args.optuna_fold}"
    storage = resolve_optuna_storage(args)
    sampler, pruner = build_optuna_components(args)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=args.optuna_direction,
        load_if_exists=True,
    )
    study.optimize(_objective_factory(args), n_trials=args.optuna_trials)

    output_dir = os.path.join(args.results_dir, "optuna", study_name)
    save_study_artifacts(study, output_dir)
    print(f"Optuna study finished. Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    start = timer()
    args = _process_args()
    args_for_study = deepcopy(args)
    main(args_for_study)
    end = timer()
    print("finished optuna tuning!")
    print("end script")
    print("Script Time: %f seconds" % (end - start))
