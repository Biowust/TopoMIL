import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloaders.data_utils import dataset_builder
from models.mil_ss import TopoMIL

from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloaders.data_utils import dataset_builder
from models.mil_ss import TopoMIL
import os
import matplotlib.pyplot as plt
from plot_distribution import plot_instance_importance_distribution



def set_deterministic_mode(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        except:
            pass
def evaluate(epoch, model, loader, args=None, set='val', show=False, visualize_dir=None):
    ds = loader.dataset
    if hasattr(ds, 'dataset'):
        base_ds, indices = ds.dataset, ds.indices
        if hasattr(base_ds, 'labels'):
            labels = [base_ds.labels[i] for i in indices]
        else:
            labels = [base_ds[i][1] for i in indices]
    else:
        labels = list(ds.labels) if hasattr(ds, 'labels') else [ds[i][1] for i in range(len(ds))]
    def _scalar(y):
        return y.item() if hasattr(y, 'item') else int(y)
    n0 = sum(1 for y in labels if _scalar(y) == 0)
    n1 = sum(1 for y in labels if _scalar(y) == 1)
    print(f">>> Validation Labels Check: 0s={n0}, 1s={n1}")

    model.eval()
    loss_meter = Meter()
    all_probs = []
    all_targets = []
    real_pos_scores = None
    real_neg_scores = None
    real_hard_scores = None
    max_pos_prob = -1.0
    min_neg_prob = 2.0
    min_diff_from_half = 1.0

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 4:
                data, target, _, _ = batch_data
            else:
                data, target, _ = batch_data
            
            data = data.cuda()
            labels = target.cuda().long().reshape(-1)
            label_val = labels.item()
            
            logits, attn_dict = model(data, return_attn=True)
            loss = F.cross_entropy(logits, labels)
            loss_meter.update(loss.item())
            
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            all_probs.append(prob)
            all_targets.append(label_val)
            current_patch_scores = attn_dict['patch_scores'][0].cpu().numpy()
            if label_val == 1 and prob > max_pos_prob:
                max_pos_prob = prob
                real_pos_scores = current_patch_scores
                
            elif label_val == 0 and prob < min_neg_prob:
                min_neg_prob = prob
                real_neg_scores = current_patch_scores
                
            diff_from_half = abs(prob - 0.5)
            if diff_from_half < min_diff_from_half:
                min_diff_from_half = diff_from_half
                real_hard_scores = current_patch_scores

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    from sklearn.metrics import (
        roc_auc_score, 
        precision_recall_curve, 
        accuracy_score, 
        average_precision_score 
    )
    
    auc_val = roc_auc_score(all_targets, all_probs)
    
    pr_auc_val = average_precision_score(all_targets, all_probs)
    
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    preds = (all_probs > best_thr).astype(int)
    
    acc = accuracy_score(all_targets, preds)
    precision = precisions[best_idx]
    recall = recalls[best_idx]
    f1 = f1_scores[best_idx]
    
    if show:
        print(f'\n[Epoch {epoch}] Best Thr: {best_thr:.4f} | PR-AUC: {pr_auc_val:.4f}')
        print(f'Acc: {acc:.4f} | AUC: {auc_val:.4f} | Recall: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f}')

    if visualize_dir is not None:
        if real_pos_scores is not None and real_neg_scores is not None and real_hard_scores is not None:
            plot_instance_importance_distribution(
                pos_scores=real_pos_scores, 
                neg_scores=real_neg_scores, 
                hard_scores=real_hard_scores, 
                k=400,
                save_dir=visualize_dir
            )
        else:
            print("Warning: Unable to collect all three typical slices, skipping distribution plot.")
    return loss_meter.avg(), acc, auc_val, best_thr, pr_auc_val, precision, recall, f1, 0


from collections import OrderedDict

def test_main(model, args, thrs=0.5, best_model=None):
    if isinstance(model, nn.DataParallel):
        model = model.module

    seed = getattr(args, 'seed', 3407)
    Dataset  = dataset_builder(args)
    lib_root = args.data_dir
    testset  = Dataset(root=lib_root, mode='test')

    loader   = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    
    if best_model is None:
        ckpt_path = args.best_model_path
    else:
        ckpt_path = best_model
        
    print(f'>>> Loading Checkpoint from: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(">>> Model weights loaded successfully (Strict Mode)!")
    except RuntimeError as e:
        print(f">>> Warning: Strict loading failed, using strict=False. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    best_thrs_from_train = checkpoint.get('best_threshold_f1', checkpoint.get('best_threshold', thrs))
    print(f'Training Best Threshold (Reference): {best_thrs_from_train:.4f}')
    
    model.eval()
    vis_dir = os.path.join(args.save_path, 'distribution_plots')
    os.makedirs(vis_dir, exist_ok=True)
    val_result = evaluate("best", model, loader, args, set='test', show=False, visualize_dir=vis_dir)
    
    test_loss = val_result[0]
    test_acc  = val_result[1]
    test_auc  = val_result[2]
    best_thr  = val_result[3] 
    test_pr_auc = val_result[4]
    test_prec = val_result[5]
    test_rec  = val_result[6]
    test_f1   = val_result[7]
    
    print(f'[test] epo:{"best":>3} | acc: {by(test_acc)} | auc: {by(test_auc)} | aupr: {by(test_pr_auc)}')
    print(f'[test] Optimal Thr: {by(best_thr)} | precision: {by(test_prec)}|recall:{by(test_rec)}|f1:{by(test_f1)}')

    return test_acc, test_auc, test_pr_auc, test_prec, test_rec, test_f1

if __name__ == '__main__':

    set_deterministic_mode(3407)
    args = setup_run(arg_mode='test')
    ''' define model '''
    model = TopoMIL(args).cuda()
    test_acc, test_auc, test_pr_auc, test_prec, test_rec, test_f1 = test_main(model, args, args.thres,args.best_model_path)
                     
    csv_path = os.path.join(args.save_path.split(args.extra_dir)[0], f'results_{args.data_name}_test.csv')
    if os.path.exists(csv_path):
        fp = open(csv_path, 'a')
    else:
        fp = open(csv_path, 'w')
        fp.write('method,acc,auc,threshold\n')
    method_name = args.model_name
    fp.write(f'{method_name},{0.01*test_acc:.4f},{0.01*test_auc:.4f},{args.thres:.3f}\n')
    fp.close()
    print()