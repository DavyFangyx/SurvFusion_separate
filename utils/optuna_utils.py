import copy
import os
from pathlib import Path


def ensure_optuna_available():
    try:
        import optuna  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna is not installed in the current environment. "
            "Install it first, e.g. `python -m pip install optuna`."
        ) from exc


def build_optuna_components(args):
    import optuna

    if args.optuna_sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif args.optuna_sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Optuna sampler `{args.optuna_sampler}`.")

    if args.optuna_pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.optuna_n_startup_trials,
            n_warmup_steps=args.optuna_n_warmup_steps,
            interval_steps=1,
        )
    elif args.optuna_pruner == "none":
        pruner = optuna.pruners.NopPruner()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Optuna pruner `{args.optuna_pruner}`.")

    return sampler, pruner


def resolve_optuna_storage(args):
    if args.optuna_storage:
        return args.optuna_storage

    base_results = Path(args.results_dir)
    storage_dir = base_results / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = args.optuna_study_name or f"{args.study}_{args.modality}_{args.poe_variant}_fold{args.optuna_fold}"
    return f"sqlite:///{(storage_dir / f'{study_name}.db').resolve()}"


def sample_survtri_poe_vae_model_c(trial, args):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-6, 1e-2, log=True),
        "poe_surv_lambda": trial.suggest_float("poe_surv_lambda", 1e-2, 1e1, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 1e-2, 2.0, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
    }
    return params


def sample_survtri_poe_vae_model_a(trial, args):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "lr_stage1": trial.suggest_float("lr_stage1", 1e-5, 5e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-6, 1e-2, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 1e-2, 2.0, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
    }
    return params


def sample_survtri_poe_vae_model_b(trial, args):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "lr_stage1": trial.suggest_float("lr_stage1", 1e-5, 5e-4, log=True),
        "reg": trial.suggest_float("reg", 1e-6, 1e-2, log=True),
        "poe_beta_target": trial.suggest_float("poe_beta_target", 1e-2, 2.0, log=True),
        "poe_modality_dropout": trial.suggest_float("poe_modality_dropout", 0.0, 0.4),
        "poe_mmhid": trial.suggest_categorical("poe_mmhid", [128, 256, 512]),
        "poe_decoder_hidden_dim": trial.suggest_categorical("poe_decoder_hidden_dim", [256, 512, 768]),
    }
    return params


def sample_survtri_poe_vae_trial(trial, args):
    if args.poe_variant == "A":
        return sample_survtri_poe_vae_model_a(trial, args)
    if args.poe_variant == "B":
        return sample_survtri_poe_vae_model_b(trial, args)
    if args.poe_variant == "C":
        return sample_survtri_poe_vae_model_c(trial, args)
    raise ValueError(f"Unsupported poe_variant `{args.poe_variant}` for Optuna.")


def build_trial_args(base_args, trial, sampled_params, pruned_exception_cls):
    trial_args = copy.deepcopy(base_args)
    for key, value in sampled_params.items():
        setattr(trial_args, key, value)

    trial_args.run_name = f"{base_args.run_name}_trial_{trial.number:04d}"
    trial_args.exp_group = f"{base_args.exp_group}_optuna"
    trial_args.use_optuna = True
    trial_args.optuna_trial = trial
    trial_args.optuna_pruned_exception = pruned_exception_cls
    trial_args.save_best_from_epoch = 0
    return trial_args


def save_study_artifacts(study, output_dir):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_trial.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_value: {study.best_value}\n")
        f.write(f"best_trial_number: {study.best_trial.number}\n")
        f.write("best_params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
