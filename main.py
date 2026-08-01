#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val_test
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args
from warnings import simplefilter

simplefilter(action="ignore",category=FutureWarning)

def _write_filter_log(args):
    summary = getattr(args.dataset_factory, "wsi_filter_summary", None)
    removed_rows = getattr(args.dataset_factory, "wsi_filter_removed_rows", None)
    if not summary:
        return

    out_path = os.path.join(args.results_dir, "filter.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WSI filter summary\n")
        f.write(f"study: {args.study}\n")
        f.write(f"modality: {args.modality}\n")
        f.write(f"data_root_dir: {summary.get('data_dir', 'N/A')}\n")
        f.write(f"total_cases_before: {summary.get('total_cases_before', 0)}\n")
        f.write(f"kept_cases: {summary.get('kept_cases', 0)}\n")
        f.write(f"removed_cases: {summary.get('removed_cases', 0)}\n")
        f.write("\n")
        f.write("Removed cases\n")

        if not removed_rows:
            f.write("None\n")
            return

        for row in removed_rows:
            missing_slide_ids = ", ".join(row["missing_slide_ids"])
            f.write(
                f"{row['case_id']}\t{subtype_label(row['oncotree_code'])}\tmissing_slides={missing_slide_ids}\n"
            )

def subtype_label(oncotree_code):
    return oncotree_code if oncotree_code not in (None, "", "nan") else "N/A"


def _init_wandb_run(args, fold):
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError("wandb is not installed, but wandb logging was requested.")

    run_name = f"{args.run_name}_fold{fold}"
    config = dict(vars(args))
    for key in ["dataset_factory", "wandb_run", "optuna_trial", "optuna_pruned_exception"]:
        config.pop(key, None)

    wandb_root = os.path.join(args.results_dir, "wandb")
    wandb_cache = os.path.join(wandb_root, "cache")
    os.makedirs(wandb_cache, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_root
    os.environ["WANDB_CACHE_DIR"] = wandb_cache
    settings = wandb.Settings(
        mode=args.wandb_mode,
        root_dir=wandb_root,
        start_method="thread",
        symlink=False,
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=f"{args.exp_group}/{args.modality}",
        name=run_name,
        dir=wandb_root,
        config=config,
        reinit=True,
        tags=[args.modality, f"poe_{getattr(args, 'poe_variant', 'na')}", f"fold_{fold}"],
        settings=settings,
    )
    return run

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    # folds = [4]
    
    #----> storing the val and test cindex for 5 fold cv
    all_test_cindex = []
    all_test_cindex_ipcw = []
    all_test_BS = []
    all_test_IBS = []
    all_test_iauc = []
    all_test_iauc_list = []
    all_test_loss = []

    for i in folds:
        args.wandb_run = _init_wandb_run(args, i)
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train/val/test datasets for fold {}".format(i))

        results, (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss), attn_matrix = _train_val_test(datasets, i, args)

        all_test_cindex.append(test_cindex)
        all_test_cindex_ipcw.append(test_cindex_ipcw)
        all_test_BS.append(test_BS)
        all_test_IBS.append(test_IBS)
        all_test_iauc.append(test_iauc)
        all_test_iauc_list.append(test_iauc_list)
        all_test_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        _save_pkl(filename, results)

        if args.wandb_run is not None:
            wandb.finish()
            args.wandb_run = None
    
    final_df = pd.DataFrame({
        'test_cindex': all_test_cindex,
        'test_cindex_ipcw': all_test_cindex_ipcw,
        'test_IBS': all_test_IBS,
        'test_iauc': all_test_iauc,
        'test_iauc_list': all_test_iauc_list,
        "test_loss": all_test_loss,
        'test_BS': all_test_BS,
    })

    if len(folds) != args.k:
        save_name = 'test_result_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'test_result.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
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
        is_mcat = True if args.modality == "mcat" else False,
        is_survpath = True if args.modality == "survpath" else False,
        is_survpgc = True if args.modality == "survpgc" else False,
        is_survpgc_f = True if args.modality in (
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
        is_survpc = True if args.modality == "survpc" else False,
        is_survpc_f = True if args.modality == "survpc_f" else False,
        type_of_pathway=args.type_of_path)

    _write_filter_log(args)


    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
