import os,sys
from tqdm import tqdm
import time
import wandb
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math,torch
import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, setup_run, by, load_model,measure_model
from models.dataloaders.data_utils import dataset_builder
from models.dataloaders.samplers import CategoriesSampler
from models.mil_ss import TopoMIL
from test import test_main, evaluate

from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.scheduler.cosine_lr import CosineLRScheduler
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

def plot_loss_curves(train_loss, val_loss, save_path, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
    plt.title(f'Loss Curves (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def plot_accuracy_curves(train_acc, val_acc, save_path, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Val Acc')
    plt.title(f'Accuracy Curves (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'acc_curve.png'))
    plt.close()

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    loss_meter = Meter()
    acc_meter  = Meter()

    if hasattr(loader.dataset, 'dataset'):
        original_dataset = loader.dataset.dataset
    else:
        original_dataset = loader.dataset

    if hasattr(original_dataset, 'count_dict'):
        ce_weight = [i for i in original_dataset.count_dict.values()]
        ce_weight = 1. / (torch.tensor(ce_weight, dtype=torch.float) + 1e-6) 
        ce_weight = ce_weight / ce_weight.mean()
        ce_weight = ce_weight.cuda()
    else:
        ce_weight = None 

    focal_loss_func = FocalLoss(alpha=ce_weight, gamma=2.0).cuda()
    
    for i, (data, labels, _, zero_idx) in enumerate(loader):
        data, labels = data.cuda(), labels.cuda().long()
        
        optimizer.zero_grad()
        
        if args.data_name == 'cm16' and args.dataset == 'cm16':
            data = F.dropout(data, p=0.20)
        
        logits = model(data) 

        # loss = focal_loss_func(logits, labels)
        loss = F.cross_entropy(logits, labels, weight=ce_weight, label_smoothing=0.1)
        # loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        
        acc = compute_accuracy(logits, labels)
        loss_meter.update(loss.item())
        acc_meter.update(acc)

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg():.4f}, Acc: {acc_meter.avg():.2f}%")
        
    return loss_meter.avg(), acc_meter.avg(), acc_meter.std()

def train_main(args):
    Dataset  = dataset_builder(args)
    lib_root = args.data_dir
    trainset = Dataset(root=lib_root, mode='train', batch=True)
   
    if args.data_name == 'msi':
        valset = Dataset(root=lib_root, mode='val')
        if hasattr(trainset, 'libs') and hasattr(valset, 'libs'):
            def _slide_id(p):
                return os.path.splitext(os.path.basename(p))[0]
            train_slide_names = {_slide_id(p) for p in trainset.libs}
            val_slide_names = {_slide_id(p) for p in valset.libs}
            overlap = train_slide_names.intersection(val_slide_names)
            print(f">>> [MSI] Train slides: {len(train_slide_names)}, Val slides: {len(val_slide_names)}, Overlap: {len(overlap)} (must be 0)")
            if len(overlap) > 0:
                raise RuntimeError(f"Data leakage: {len(overlap)} slide(s) in both train and val.")
    else:
        val_ratio = 0.2
        import random
        from torch.utils.data import Subset

        def _slide_id(path):
            return os.path.splitext(os.path.basename(path))[0]

        original_dataset = trainset
        if hasattr(original_dataset, 'libs'):
            slide_ids = [_slide_id(p) for p in original_dataset.libs]
            unique_slide_ids = sorted(set(slide_ids))
            random.seed(args.seed)
            random.shuffle(unique_slide_ids)
            n_val = max(1, int(len(unique_slide_ids) * val_ratio))
            val_slide_ids = set(unique_slide_ids[:n_val])
            train_slide_ids = set(unique_slide_ids[n_val:])
            train_indices = [i for i in range(len(original_dataset.libs)) if _slide_id(original_dataset.libs[i]) in train_slide_ids]
            val_indices = [i for i in range(len(original_dataset.libs)) if _slide_id(original_dataset.libs[i]) in val_slide_ids]
        else:
            total_size = len(trainset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            random.seed(args.seed)
            indices = list(range(total_size))
            random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

        trainset = Subset(original_dataset, train_indices)
        valset = Subset(original_dataset, val_indices)

        train_slide_names = set()
        val_slide_names = set()
        if hasattr(original_dataset, 'libs'):
            train_slide_names = {_slide_id(original_dataset.libs[i]) for i in train_indices}
            val_slide_names = {_slide_id(original_dataset.libs[i]) for i in val_indices}
        overlap = train_slide_names.intersection(val_slide_names)
        print(f">>> Train slides: {len(train_slide_names)}, Val slides: {len(val_slide_names)}, Overlap: {len(overlap)} (must be 0)")
        if len(overlap) > 0:
            raise RuntimeError(f"Data leakage: {len(overlap)} slide(s) appear in both train and val. Split must be by Slide/Patient ID.")

        original_trainset = original_dataset
        if hasattr(original_trainset, 'labels'):
            subset_labels = [original_trainset.labels[i] for i in train_indices]
        else:
            subset_labels = [original_trainset[i][1] for i in train_indices]

        train_sampler = CategoriesSampler(subset_labels, n_batch=len(trainset), n_cls=args.num_class, n_per=1)

    dl_workers = max(4, min(16, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=dl_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=dl_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
     
    set_seed(args.seed)
       
    if args.model_name == 'TopoMIL':
        model = TopoMIL(args).cuda()
    else:
        raise ValueError('Model not found')

    if hasattr(args, 'use_mixup') and args.use_mixup:
        model.use_mixup = True
        print(">>> Mixup Enabled in Training (Auto-disable FeatMag during mixed batches) <<<")

    model = nn.DataParallel(model, device_ids=args.device_ids)
    
    if not args.no_wandb and wandb.run is not None:
        wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.wd)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc = -1.0
    optimal_thresholds = []
    set_seed(args.seed)
    
    print('Training :::::\n')
    hist_train_losses, hist_val_losses = [], []
    hist_train_accs,   hist_val_accs   = [], []
    best_model_path = None
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.max_epoch, 
    eta_min=1e-6
)
    for epoch in tqdm(range(1, args.max_epoch + 1)):
        train_loss, train_acc, _ = train(epoch, model, train_loader, optimizer, args)
        val_loss, val_acc, val_auc, val_thrs, val_pr_auc, val_prec, val_rec, val_f1, val_f1_opt_thrs = evaluate(epoch, model, val_loader, args, set='val', show=True)

        scheduler.step()

        if not args.no_wandb and wandb.run is not None:
          wandb.log({
                'train/loss': train_loss,
                'train/acc':  train_acc,
                'val/loss':   val_loss,
                'val/acc':    val_acc,
                'val/auc':    val_auc,
            }, step=epoch)

        if val_acc > max_acc: 
            optimal_thresholds.append(val_thrs)
            max_acc = val_acc
            ckpt = {
                'params': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'best_threshold': float(val_thrs), 
                'epoch': int(epoch), 
                'args': vars(args),
                'best_threshold_f1': float(val_f1_opt_thrs),
                'best_f1': float(val_f1),
                'best_prec': float(val_prec),
                'best_rec': float(val_rec),
                'best_pr_auc': float(val_pr_auc),
                'best_acc': float(val_acc),
                'best_auc': float(val_auc),
                'best_epoch': int(epoch)
                }
            best_model_path = os.path.join(args.save_path, 'max_acc.pth')
            
            torch.save(ckpt, best_model_path)
            if val_acc > 0.83:
                model_path = os.path.join(args.save_path, f'epoch_{epoch}_{val_acc:.3f}_{val_auc:.3f}.pth')
                torch.save(ckpt, model_path)

        hist_train_losses.append(train_loss)
        hist_val_losses.append(val_loss)
        hist_train_accs.append(train_acc)
        hist_val_accs.append(val_acc)
        
        plot_loss_curves(hist_train_losses, hist_val_losses, args.save_path, epoch)
        plot_accuracy_curves(hist_train_accs, hist_val_accs, args.save_path, epoch)
        
        # lr_scheduler.step()
        scheduler.step()
    
    print(f"Optimal Thresholds history: {optimal_thresholds}")
    return model, optimal_thresholds[-1] if optimal_thresholds else 0.5, best_model_path

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # alpha 可以是 float (正样本权重) 或 list (各类权重)
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([1-alpha, alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # 获取对应的 alpha
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = focal_loss * at

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss
if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    
    model, thrs,best_model_path = train_main(args)
    print(f'Best Threshold ::: {thrs:.3f}')
    test_acc, test_auc, test_pr_auc, test_prec, test_rec, test_f1 = test_main(model, args, thrs,best_model_path)

    csv_path = os.path.join(args.save_path.split(args.extra_dir)[0], f'results_{args.data_name}.csv')
    if os.path.exists(csv_path):
        fp = open(csv_path, 'a')
    else:
        fp = open(csv_path, 'w')
        fp.write('method,acc,auc,threshold\n')

    method_name = args.model_name 
    fp.write(f'{method_name},{test_acc:.4f},{test_auc:.4f},{thrs:.4f},{test_f1:.4f},{test_prec:.4f},{test_rec:.4f}\n')
    fp.close()
    print()

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/auc': test_auc})
        