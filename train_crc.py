import os, sys
from tqdm import tqdm
import time
import wandb
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader, Subset

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, setup_run, by, load_model, measure_model, plot_cv_roc_pr_curves
from models.dataloaders.data_utils import dataset_builder
from models.dataloaders.samplers import CategoriesSampler
from models.mil_ss import TopoMIL
from test import test_main, evaluate
from sklearn.metrics import roc_curve, auc, average_precision_score
import pickle
import json
import csv
class FeatMag(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        
    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)
        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um/w_scale

def plot_loss_curves(train_loss, val_loss, save_path, epoch, fold_idx):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
    plt.title(f'Fold {fold_idx} Loss Curves (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'loss_curve_fold{fold_idx}.png'))
    plt.close()

def plot_accuracy_curves(train_acc, val_acc, save_path, epoch, fold_idx):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Val Acc')
    plt.title(f'Fold {fold_idx} Accuracy Curves (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'acc_curve_fold{fold_idx}.png'))
    plt.close()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        f_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return f_loss.mean()
        else:
            return f_loss.sum()

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    loss_meter = Meter()
    acc_meter  = Meter()

    if hasattr(loader.dataset, 'dataset'): # Handle Subset
        original_dataset = loader.dataset.dataset
    else:
        original_dataset = loader.dataset

    ce_weight = None
    if hasattr(original_dataset, 'count_dict'):
        ce_weight = [i for i in original_dataset.count_dict.values()]
        if 0 in ce_weight: ce_weight = [c + 1 for c in ce_weight]
        ce_weight = 1. / torch.tensor(ce_weight, dtype=torch.float)
        ce_weight = ce_weight.cuda()

    for i, (data, labels, _, zero_idx) in enumerate(loader):
        data, labels = data.cuda(), labels.cuda().long()
        
        # noise = torch.randn_like(data) * 0.01
        # data = data + noise
        
        optimizer.zero_grad()
        
        logits = model(data) 

        loss = F.cross_entropy(logits, labels, weight=ce_weight, label_smoothing=0.1)

        loss.backward()
        optimizer.step()
        
        acc = compute_accuracy(logits, labels)
        loss_meter.update(loss.item())
        acc_meter.update(acc)

    if epoch % 5 == 0:
        print(f"    Epoch [{epoch}] Train Loss: {loss_meter.avg():.4f}, Acc: {acc_meter.avg():.2f}%")
        
    return loss_meter.avg(), acc_meter.avg(), acc_meter.std()

from collections import defaultdict

def train_main(args):
    print(">>> Applying CRC Specific Settings (Scheme A + C)...")
    args.num_feats = 768   
    k_folds = 5            

    Dataset = dataset_builder(args)
    lib_root = args.data_dir
    
    full_trainset = Dataset(root=lib_root, mode='train', batch=True)
    
    if hasattr(full_trainset, 'path_labels'):
        all_labels = full_trainset.path_labels
    elif hasattr(full_trainset, 'labels'):
        all_labels = full_trainset.labels
    else:
        all_labels = []
        for i in range(len(full_trainset)):
            _, y, _ = full_trainset[i]
            all_labels.append(int(y))
    all_labels = np.array(all_labels)
    
    # skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    split_pkl_path = "/mnt/data/ljc/outputs/TopoMIL/crc/2026_March_16_15h_05m_mamba/crc_5fold_splits_seed1.pkl"
    with open(split_pkl_path, "rb") as f:
        split_pack = pickle.load(f)

    splits = split_pack["splits"]
    assert len(splits) == k_folds, f"Split pkl K={len(splits)} != expected {k_folds}"

    # split_pkl_path = os.path.join(
    #     args.save_path, f"crc_{k_folds}fold_splits_seed{args.seed}.pkl"
    # )
    if not os.path.exists(split_pkl_path):
        fold_info = []
        for i, (tr_idx, va_idx) in enumerate(splits, start=1):
            va_labels = all_labels[va_idx]
            fold_info.append({
                "fold": i,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "val_pos": int((va_labels == 1).sum()),
                "val_neg": int((va_labels == 0).sum()),
            })

        with open(split_pkl_path, "wb") as f:
            pickle.dump(
                {
                    "dataset": "crc",
                    "k_folds": k_folds,
                    "seed": args.seed,
                    "splits": [(tr_idx, va_idx) for (tr_idx, va_idx) in splits],
                    "fold_info": fold_info,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[Saved] CV splits to: {split_pkl_path}")
    else:
        print(f"[Found] CV splits already exist: {split_pkl_path}")
    
    cv_metrics = defaultdict(list)

    oof_y_true_folds = []
    oof_y_score_folds = []

    fold_curve_auc_list = []
    fold_curve_ap_list = []

    fold_records = []

    global_best_auc = -1.0
    global_best_model_path = None
    
    print(f"\n{'='*20} Starting {k_folds}-Fold Cross Validation {'='*20}")
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        fold_best_model_path = None
        print(f"\n>>> Fold {fold_idx + 1}/{k_folds}")
        
        train_subset = Subset(full_trainset, train_indices)
        val_subset = Subset(full_trainset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        
        set_seed(args.seed + fold_idx) 
        model = TopoMIL(args).cuda()
        model = nn.DataParallel(model, device_ids=args.device_ids)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
        
        fold_max_auc = -1.0
        fold_best_epoch = -1
        fold_best_metrics = {} 
        
        for epoch in tqdm(range(1, args.max_epoch + 1), desc=f'Fold {fold_idx+1}', leave=False):
            train_loss, train_acc, _ = train(epoch, model, train_loader, optimizer, args)
            
            val_results = evaluate(epoch, model, val_loader, args, set='val', show=False)
            
            v_loss  = val_results[0]
            v_acc   = val_results[1]
            v_auc   = val_results[2]
            v_thrs  = val_results[3]
            v_prauc = val_results[4]
            v_prec  = val_results[5]
            v_rec   = val_results[6]
            v_f1    = val_results[7]
            
            scheduler.step()
            
            if v_auc > fold_max_auc:
                fold_max_auc = v_auc
                fold_best_epoch = epoch
                
                fold_best_metrics = {
                    'acc': v_acc,
                    'auc': v_auc,
                    'pr_auc': v_prauc,
                    'precision': v_prec,
                    'recall': v_rec,
                    'f1': v_f1,
                    'threshold': v_thrs
                }
                
                ckpt_name = f'fold_{fold_idx+1}_best.pth'
                save_path = os.path.join(args.save_path, ckpt_name)
                torch.save(model.state_dict(), save_path)
                fold_best_model_path = save_path

                if v_auc > global_best_auc:
                    global_best_auc = v_auc
                    global_best_model_path = save_path
                

        print(f"    Fold {fold_idx+1} Best: AUC={fold_best_metrics['auc']:.4f} | Acc={fold_best_metrics['acc']:.4f}")
        val_labels = all_labels[val_indices]
        assert fold_best_model_path is not None, f"Fold {fold_idx+1} has no best checkpoint!"

        state = torch.load(fold_best_model_path, map_location="cuda")
        model.load_state_dict(state, strict=True)

        y_true_fold, y_score_fold = collect_fold_scores(model, val_loader)

        oof_y_true_folds.append(y_true_fold)
        oof_y_score_folds.append(y_score_fold)

        fpr, tpr, _ = roc_curve(y_true_fold, y_score_fold, pos_label=1)
        fold_auc_curve = auc(fpr, tpr)
        fold_ap_curve  = average_precision_score(y_true_fold, y_score_fold)

        fold_curve_auc_list.append(fold_auc_curve)
        fold_curve_ap_list.append(fold_ap_curve)

        val_labels = all_labels[val_indices]
        fold_records.append({
            "fold": int(fold_idx + 1),
            "best_epoch": int(fold_best_epoch),
            "ckpt_path": str(fold_best_model_path),
            "val_size": int(len(val_indices)),
            "val_pos": int((val_labels == 1).sum()),
            "val_neg": int((val_labels == 0).sum()),

            "acc": float(fold_best_metrics["acc"]),
            "auc": float(fold_best_metrics["auc"]),
            "pr_auc": float(fold_best_metrics["pr_auc"]),
            "precision": float(fold_best_metrics["precision"]),
            "recall": float(fold_best_metrics["recall"]),
            "f1": float(fold_best_metrics["f1"]),
            "threshold": float(fold_best_metrics["threshold"]),

            "curve_auc": float(fold_auc_curve),
            "curve_ap": float(fold_ap_curve),
        })

        print(f"    [Fold {fold_idx+1}] OOF Curve Metrics (best ckpt): "
            f"ROC-AUC={fold_auc_curve:.4f}, AP(AUPR)={fold_ap_curve:.4f}")
        for k, v in fold_best_metrics.items():
            cv_metrics[k].append(v)
    curve_auc_mean = float(np.mean([r["curve_auc"] for r in fold_records]))
    curve_auc_std  = float(np.std([r["curve_auc"] for r in fold_records]))
    curve_ap_mean  = float(np.mean([r["curve_ap"] for r in fold_records]))
    curve_ap_std   = float(np.std([r["curve_ap"] for r in fold_records]))
    print(f"\n{'='*20} Final CV Summary (Mean ± Std) {'='*20}")
    print("\n" + "="*20 + " Fold-wise ROC-AUC / AP (from OOF scores) " + "="*20)
    for i, (a, p) in enumerate(zip(fold_curve_auc_list, fold_curve_ap_list), start=1):
        print(f"Fold {i}: AUC={a:.4f}, AP={p:.4f}")

    print(f"\nAUC mean±std = {np.mean(fold_curve_auc_list):.4f} ± {np.std(fold_curve_auc_list):.4f}")
    print(f"AP  mean±std = {np.mean(fold_curve_ap_list):.4f} ± {np.std(fold_curve_ap_list):.4f}")
    print("="*72 + "\n")

    oof_pkl_path = os.path.join(args.save_path, "crc_oof_scores.pkl")
    with open(oof_pkl_path, "wb") as f:
        pickle.dump(
            {
                "dataset": "crc",
                "k_folds": k_folds,
                "seed": args.seed,
                "y_true_folds": oof_y_true_folds,
                "y_score_folds": oof_y_score_folds,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    plot_cv_roc_pr_curves(
        oof_y_true_folds,
        oof_y_score_folds,
        dataset_name="crc",
        save_dir=args.save_path,
        split_name=f"5-fold CV (OOF)",
        show_band=True,
        plot_pooled=False
    )
    
    summary_txt = []
    headers = ["Metric", "Mean", "Std", "Min", "Max"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}")
    print("-" * 60)
    
    summary_txt.append(",".join(headers)) # CSV header
    
    final_thrs = 0.5
    
    for metric, values in cv_metrics.items():
        mean_v = np.mean(values)
        std_v  = np.std(values)
        min_v  = np.min(values)
        max_v  = np.max(values)
        
        print(f"{metric:<15} {mean_v:.4f}     {std_v:.4f}     {min_v:.4f}     {max_v:.4f}")
        
        summary_txt.append(f"{metric},{mean_v:.4f},{std_v:.4f},{min_v:.4f},{max_v:.4f}")
        
        if metric == 'threshold':
            final_thrs = mean_v

    print(f"{'='*60}\n")
    results_json_path = os.path.join(args.save_path, "crc_cv_fold_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "crc",
                "k_folds": k_folds,
                "seed": args.seed,
                "global_best_auc": float(global_best_auc),
                "global_best_model_path": str(global_best_model_path),
                "global_best_threshold": float(final_thrs),
                "fold_records": fold_records,
                "summary": {
                    "curve_auc_mean": float(np.mean(fold_curve_auc_list)),
                    "curve_auc_std": float(np.std(fold_curve_auc_list)),
                    "curve_ap_mean": float(np.mean(fold_curve_ap_list)),
                    "curve_ap_std": float(np.std(fold_curve_ap_list)),
                }
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"[Saved] Fold results JSON to: {results_json_path}")
    results_csv_path = os.path.join(args.save_path, "crc_cv_fold_results.csv")
    csv_cols = [
        "fold","best_epoch","val_size","val_pos","val_neg","acc","auc","pr_auc",
        "precision","recall","f1","threshold","curve_auc","curve_ap","ckpt_path"
    ]
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in fold_records:
            writer.writerow({k: r.get(k, "") for k in csv_cols})
    print(f"[Saved] Fold results CSV to: {results_csv_path}")
    with open(os.path.join(args.save_path, 'cv_full_metrics.csv'), 'w') as f:
        f.write("\n".join(summary_txt))

    
    return model, final_thrs, global_best_model_path

@torch.no_grad()
def collect_fold_scores(model, loader):
    """
    返回:
      y_true: np.ndarray shape [N]
      y_score: np.ndarray shape [N]  (正类概率)
    """
    model.eval()
    ys, ps = [], []

    for batch in loader:
        data, labels = batch[0], batch[1]
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long()

        logits = model(data)  # [B,2]
        prob_pos = torch.softmax(logits, dim=1)[:, 1]

        ys.append(labels.detach().cpu().numpy())
        ps.append(prob_pos.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return y_true, y_score

if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    
    model, thrs, best_model_path = train_main(args)
    
    print(f'Global Best Threshold ::: {thrs:.3f}')
    
    print("Training and Cross-Validation Complete!")
    print("Please check 'cv_summary.txt' in your output folder for the final results.")