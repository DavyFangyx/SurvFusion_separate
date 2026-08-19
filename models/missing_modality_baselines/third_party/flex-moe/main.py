import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import os
import torch
import numpy as np
import argparse
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from tqdm import trange
from lifelines.utils import concordance_index
from models import FlexMoE
from utils import seed_everything, setup_logger
from utils.loss_func import CoxSurvLoss
from models.missing_modality_baselines.result_utils import build_results_root, build_seed_dir, write_experiment_file
from data import load_and_preprocess_data, load_and_preprocess_survpgc_data, create_loaders, create_survpgc_loaders
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--study', type=str, default='tcga_lihc')
    parser.add_argument('--modality', type=str, default='IGCB') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--exp_group', type=str, default='FlexMoE')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--preprocessed', type=str2bool, default=True) # Whether to use preprocessed image modality
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=16) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=4) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--split_csv', type=str, default=None, help='Path to train/val/test split csv')
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=3)

    return parser.parse_known_args()


def _cox_cindex(times, risks, censors):
    times = np.asarray(times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    censors = np.asarray(censors, dtype=float)
    return float(concordance_index(times, -risks, 1.0 - censors))

def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)
        
        fusion_input = []
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)

        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + gate_loss_weight * gate_loss
            loss.backward()
            optimizer.step()
        else:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses
    else:
        return all_preds, all_labels, all_probs


def run_epoch_survpgc(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0):
    all_risks = []
    all_times = []
    all_censors = []
    task_losses = []
    gate_losses = []

    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_times, batch_censors, batch_mcs, batch_observed, batch_case_ids in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_times = batch_times.to(device, non_blocking=True)
        batch_censors = batch_censors.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)

        fusion_input = []
        for modality, samples in batch_samples.items():
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim), device=device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)

        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)
        risk = outputs.reshape(-1)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_times, batch_censors)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + gate_loss_weight * gate_loss
            loss.backward()
            optimizer.step()
        else:
            all_risks.extend(risk.detach().cpu().numpy().tolist())
            all_times.extend(batch_times.detach().cpu().numpy().tolist())
            all_censors.extend(batch_censors.detach().cpu().numpy().tolist())

    if is_training:
        return task_losses, gate_losses
    return all_risks, all_times, all_censors


def train_and_evaluate(args, seed, save_path=None):
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if args.data == 'survpgc' and args.modality == 'IGCB':
        args.modality = 'WGC'
    if args.data == 'survpgc' and not args.split_csv:
        raise ValueError("`--split_csv` is required for `--data survpgc`.")
    num_modalities = len(args.modality)

    results_root = build_results_root(args, f"flex_moe__{args.modality}")
    seed_dir = build_seed_dir(results_root, seed)

    if args.data == 'adni':
        modality_dict = {'image':0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
        args.n_full_modalities = len(modality_dict)
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_data(args, modality_dict)
    elif args.data == 'survpgc':
        modality_dict = {'wsi': 0, 'genomic': 1, 'clinical': 2}
        args.n_full_modalities = len(modality_dict)
        data_dict, encoder_dict, survival_times, censors, case_ids, train_ids, valid_ids, test_ids, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_survpgc_data(args, modality_dict)
    else:
        raise ValueError(f"Unsupported data source `{args.data}`.")
    
    if args.data == 'survpgc':
        train_loader, train_loader_shuffle, val_loader, test_loader = create_survpgc_loaders(data_dict, observed_idx_arr, survival_times, censors, case_ids, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.preprocessed, args.use_common_ids)
        output_dim = 1
        criterion = CoxSurvLoss(reduction='mean')
    else:
        train_loader, train_loader_shuffle, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.preprocessed, args.use_common_ids)
        output_dim = n_labels
        criterion = torch.nn.CrossEntropyLoss() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))

    write_experiment_file(results_root, {
        "exp_group": args.exp_group,
        "run_name": args.run_name,
        "study": args.study,
        "data": args.data,
        "modality": args.modality,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
    })

    fusion_model = FlexMoE(num_modalities, full_modality_index, args.num_patches, args.hidden_dim, output_dim, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout).to(device)
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]    
    if num_modalities > 1:
        missing_embeds = torch.nn.Parameter(torch.randn((2**num_modalities)-1, args.n_full_modalities, args.num_patches, args.hidden_dim, dtype=torch.float, device=device), requires_grad=True)
        params += [missing_embeds]

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_val_acc = 0.0
    best_val_cindex = 0.0
    best_model_me = deepcopy(missing_embeds)
    best_model_fus = deepcopy(fusion_model)
    best_model_enc = deepcopy(encoder_dict)

    if save_path is None:
        for epoch in trange(args.train_epochs):
            fusion_model.train()
            for encoder in encoder_dict.values():
                encoder.train()

            if epoch >= args.warm_up_epochs:
                train_loader_new = train_loader_shuffle
                warm_up_tag = ''
                train_epochs = args.train_epochs
            else:
                # activate modality-based sorting
                train_loader_new = train_loader
                warm_up_tag = 'Warm Up ' 
                train_epochs = args.warm_up_epochs

            ## Training
            task_losses, gate_losses = run_epoch(args, train_loader_new, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=True, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight)
            
            ## Validation
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            with torch.no_grad():
                if args.data == 'survpgc':
                    val_risks, val_times, val_censors = run_epoch_survpgc(args, val_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device)
                    val_cindex = _cox_cindex(val_times, val_risks, val_censors)
                    val_loss = float(np.mean(task_losses)) if task_losses else 0.0
                    if val_cindex > best_val_cindex:
                        print(f" [(**Best**) {warm_up_tag}Epoch {epoch+1}/{train_epochs}] Val C-index: {val_cindex:.4f}")
                        best_val_cindex = val_cindex
                        best_model_me = deepcopy(missing_embeds)
                        best_model_fus = deepcopy(fusion_model)
                        best_model_enc = deepcopy(encoder_dict)
                else:
                    val_preds, val_labels, val_probs = run_epoch(args, val_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device)
                    val_acc = accuracy_score(val_labels, val_preds)
                    val_f1 = f1_score(val_labels, val_preds, average='macro')
                    val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')

                    if val_acc > best_val_acc:
                        print(f" [(**Best**) {warm_up_tag}Epoch {epoch+1}/{train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                        best_val_acc = val_acc
                        best_val_f1 = val_f1
                        best_val_auc = val_auc
                        best_model_me = deepcopy(missing_embeds)
                        best_model_fus = deepcopy(fusion_model)
                        best_model_enc = deepcopy(encoder_dict)

                    print(f"[Seed {seed}/{args.n_runs-1}] [{warm_up_tag}Epoch {epoch+1}/{train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f} / Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")

        # Save the best model
        if args.save:
            os.makedirs(seed_dir, exist_ok=True)
            save_path = str(seed_dir / f'seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth')
            torch.save({
                'missing_embeds': best_model_me,
                'fusion_model': best_model_fus.state_dict(),
                'encoder_dict': {modality: deepcopy(encoder.state_dict()) for modality, encoder in best_model_enc.items()}
            }, save_path)

            print(f"Best model saved to {save_path}")
    
    else:
        best_model_me = missing_embeds
        best_model_fus = fusion_model
        best_model_enc = encoder_dict

        # Load the saved model onto the correct device (GPU or CPU)
        checkpoint = torch.load(save_path, map_location=device)

        # Load the models' states
        best_model_me = checkpoint['missing_embeds']
        best_model_fus.load_state_dict(checkpoint['fusion_model'])
        for modality, encoder in best_model_enc.items():
            encoder.load_state_dict(checkpoint['encoder_dict'][modality])
            encoder.to(device)
            encoder.eval()

        # Move the models to the correct device if necessary
        best_model_me.to(device)
        best_model_fus.to(device)

        ## Validation
        with torch.no_grad():
            val_preds, val_labels, val_probs = run_epoch(args, val_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
        best_val_acc = accuracy_score(val_labels, val_preds)
        best_val_f1 = f1_score(val_labels, val_preds, average='macro')
        best_val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')

    ## Test
    with torch.no_grad():
        if args.data == 'survpgc':
            test_risks, test_times, test_censors = run_epoch_survpgc(args, test_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
            test_cindex = _cox_cindex(test_times, test_risks, test_censors)
            seed_results = pd.DataFrame([{
                "seed": seed,
                "val_cindex": best_val_cindex,
                "test_cindex": test_cindex,
            }])
            seed_results.to_csv(seed_dir / "test_result.csv", index=False)
            return best_val_cindex, test_cindex
        else:
            test_preds, test_labels, test_probs = run_epoch(args, test_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
            seed_results = pd.DataFrame([{
                "seed": seed,
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
                "val_auc": best_val_auc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc,
            }])
            seed_results.to_csv(seed_dir / "test_result.csv", index=False)
            return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc

def main(args=None):
    if args is None:
        args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    val_cindices = []
    test_cindices = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'FlexMoE',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        if (not args.save) & (args.load_model):
            save_path = str(build_seed_dir(build_results_root(args, f"flex_moe__{args.modality}"), seed) / f'seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth')
        else:
            save_path = None
        result = train_and_evaluate(args, seed, save_path=save_path)
        if args.data == 'survpgc':
            val_cindex, test_cindex = result
            val_cindices.append(val_cindex)
            test_cindices.append(test_cindex)
        else:
            val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = result
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_aucs.append(test_auc)
    
    if args.data == 'survpgc':
        val_avg_cindex = float(np.mean(val_cindices))
        val_std_cindex = float(np.std(val_cindices))
        test_avg_cindex = float(np.mean(test_cindices))
        test_std_cindex = float(np.std(test_cindices))
        log_summary += f'[Val] Average C-index: {val_avg_cindex:.4f} ± {val_std_cindex:.4f} / '
        log_summary += f'[Test] Average C-index: {test_avg_cindex:.4f} ± {test_std_cindex:.4f} '
        summary_df = pd.DataFrame({
            "seed": list(seeds),
            "val_cindex": val_cindices,
            "test_cindex": test_cindices,
        })
        summary_df.to_csv(build_results_root(args, f"flex_moe__{args.modality}") / "test_result.csv", index=False)
        print(model_kwargs)
        print(f'[Val] Average C-index: {val_avg_cindex:.4f} ± {val_std_cindex:.4f}')
        print(f'[Test] Average C-index: {test_avg_cindex:.4f} ± {test_std_cindex:.4f}')
    else:
        val_avg_acc = np.mean(val_accs)*100
        val_std_acc = np.std(val_accs)*100
        val_avg_f1 = np.mean(val_f1s)*100
        val_std_f1 = np.std(val_f1s)*100
        val_avg_auc = np.mean(val_aucs)*100
        val_std_auc = np.std(val_aucs)*100

        test_avg_acc = np.mean(test_accs)*100
        test_std_acc = np.std(test_accs)*100
        test_avg_f1 = np.mean(test_f1s)*100
        test_std_f1 = np.std(test_f1s)*100
        test_avg_auc = np.mean(test_aucs)*100
        test_std_auc = np.std(test_aucs)*100

        log_summary += f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} '
        log_summary += f'[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} '
        log_summary += f'[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / '  
        log_summary += f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} '
        log_summary += f'[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} '
        log_summary += f'[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} '  

        print(model_kwargs)
        print(f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}')
        print(f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}')

    logger.info(log_summary)

if __name__ == '__main__':
    main()
