import os
from torch._C import Value
import wandb
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored
from datetime import datetime

# -*- coding: utf-8 -*-
# Python 3.8.20 compatible
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

def _interp_unique(x, y):
    """
    对 x 去重并确保递增，方便 np.interp
    x: 1D array
    y: 1D array
    """
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    return x_unique, y_unique


def plot_cv_roc_pr_curves(
    y_true_folds,
    y_score_folds,
    dataset_name,
    save_dir="./plots",
    split_name="cv_test",
    n_grid=1001,
    show_band=True,
    plot_pooled=True
):
    """
    y_true_folds: list of 1D array/list, length=K folds
    y_score_folds: list of 1D array/list, length=K folds
    说明：传入的应该是每折“test(out-of-fold)”的预测，别用train上的
    """
    assert len(y_true_folds) == len(y_score_folds), "fold 数不一致"
    K = len(y_true_folds)
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 1) ROC: mean±std + pooled
    # =========================
    mean_fpr = np.linspace(0.0, 1.0, n_grid)
    tprs = []
    aucs = []

    for k in range(K):
        y_true = np.asarray(y_true_folds[k]).astype(int)
        y_score = np.asarray(y_score_folds[k]).astype(float)

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # pooled ROC
    pooled_auc = None
    if plot_pooled:
        y_true_all = np.concatenate([np.asarray(x).astype(int) for x in y_true_folds], axis=0)
        y_score_all = np.concatenate([np.asarray(x).astype(float) for x in y_score_folds], axis=0)
        fpr_all, tpr_all, _ = roc_curve(y_true_all, y_score_all, pos_label=1)
        pooled_auc = auc(fpr_all, tpr_all)

    plt.figure(figsize=(5, 4))
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})")
    if show_band:
        lo = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
        hi = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
        plt.fill_between(mean_fpr, lo, hi, alpha=0.2, label="±1 std (TPR)")
    if plot_pooled:
        plt.plot(fpr_all, tpr_all, lw=1.5, linestyle="--", label=f"Pooled ROC (AUC={pooled_auc:.3f})")

    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name} ({split_name}, K={K})")
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, f"{dataset_name}_{split_name}_roc_cv.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()


    recall_grid = np.linspace(0.0, 1.0, n_grid)
    precisions_interp = []
    aps = []

    for k in range(K):
        y_true = np.asarray(y_true_folds[k]).astype(int)
        y_score = np.asarray(y_score_folds[k]).astype(float)

        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

        recall_u, precision_u = _interp_unique(recall, precision)
        p_i = np.interp(recall_grid, recall_u, precision_u)
        precisions_interp.append(p_i)

    precisions_interp = np.vstack(precisions_interp)
    mean_p = precisions_interp.mean(axis=0)
    std_p = precisions_interp.std(axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    pooled_ap = None
    if plot_pooled:
        precision_all, recall_all, _ = precision_recall_curve(y_true_all, y_score_all, pos_label=1)
        pooled_ap = average_precision_score(y_true_all, y_score_all)

    plt.figure(figsize=(5, 4))
    plt.plot(recall_grid, mean_p, lw=2, label=f"Mean PR (AP={mean_ap:.3f}±{std_ap:.3f})")
    if show_band:
        lo = np.clip(mean_p - std_p, 0.0, 1.0)
        hi = np.clip(mean_p + std_p, 0.0, 1.0)
        plt.fill_between(recall_grid, lo, hi, alpha=0.2, label="±1 std (Precision)")
    if plot_pooled:
        plt.plot(recall_all, precision_all, lw=1.5, linestyle="--", label=f"Pooled PR (AP={pooled_ap:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {dataset_name} ({split_name}, K={K})")
    plt.legend(loc="lower left")
    pr_path = os.path.join(save_dir, f"{dataset_name}_{split_name}_pr_cv.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    print(f"[{dataset_name} | {split_name} | K={K}]")
    print(f"  ROC mean AUC={mean_auc:.4f} ± {std_auc:.4f} | pooled AUC={pooled_auc if pooled_auc is not None else 'NA'}")
    print(f"  PR  mean AP ={mean_ap:.4f} ± {std_ap:.4f} | pooled AP ={pooled_ap if pooled_ap is not None else 'NA'}")
    print(f"  Saved ROC to: {roc_path}")
    print(f"  Saved PR  to: {pr_path}")

def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))
    print()

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    if arg_mode == 'train':
        args.save_path = args.output_dir + '/'+ args.model_name + '/'+ args.dataset + '/' +  datetime.now().strftime('%Y_%B_%d_%Hh_%Mm') + '_mamba/'
    else:
        args.save_path = args.output_dir 
    ensure_path(args.save_path)
    args.extra_dir = args.save_path + '/' +args.extra_dir
    if not args.no_wandb:
        wandb.init(project=f'mil-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.extra_dir)

    if args.dataset == 'cm16' or args.dataset == 'crc': 
        args.num_class = 2
    elif args.dataset == 'simclr': 
        args.num_class = 1
    else:
        raise ValueError('Unknown Dataset - Specify class count. ')

    return args


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

def compute_accuracy_bce(logits, labels, thr=0.5):
    pred = torch.ge(logits, thr).float()
    return (pred == labels).type(torch.float).mean().item() * 100.


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):
    model_dict = model.state_dict()
    checkpoint = torch.load(dir)
    
    # Handle both old and new checkpoint formats
    if 'params' in checkpoint:
        pretrained_dict = checkpoint['params']
    else:
        pretrained_dict = checkpoint  # Direct state_dict

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        #print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:  
        ''' Works '''
        model.load_state_dict(pretrained_dict,strict=False)

    return model


def load_model_and_optimizer(model, optimizer, dir):
    """Load both model and optimizer from a single checkpoint file"""
    checkpoint = torch.load(dir)
    
    # Load model
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer if available
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Return epoch if available
    epoch = checkpoint.get('epoch', 0)
    
    return model, optimizer, epoch


def set_seed(seed):
    if seed == 0:
        #print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        #print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def detect_grad_nan(model):
    for param in model.parameters():
        if param.requires_grad:
            if (param.grad != param.grad):
                if param.grad.float().sum() != 0:  # nan detected
                    param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow

def print_network(net, show_net=False):
    """ Print network definition"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net) if show_net else print("")
    num_params = num_params / 1000000.0
    print("----------------------------")
    print("MODEL: {:.5f}M".format(num_params))
    print("----------------------------")


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='Meta MIL')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='cm16',
                        choices=['cm16', 'simclr', 'crc'])
    parser.add_argument('-data_dir', type=str, default='datasets', help='dir of datasets')
    parser.add_argument('-data_name',type=str, default='cm16',    help='name of dataset',
    choices=['cm16','simclr', 'msi', 'crc'])
    parser.add_argument('-output_dir', type=str, default='/data/ljc/source/outputs')
    ''' about wsi-bags dataset '''
    parser.add_argument("-best_model_path", type=str, default='', help='the best model path')
    parser.add_argument("-num_feats", type=int, default=512, help='feature dimension of each instance')
    parser.add_argument("-thres",     type=float, default=0.5, help='optimal threshold for class separation')
    
    ''' about simclr '''
    parser.add_argument("-out_dim", type=int, default=256, help='output dimension of the projection head')
    
    ''' about TopoMIL '''
    parser.add_argument("-mag", type=float, default=8.48, help='margin used in the feature loss (cm16)')
    parser.add_argument('-model_name', type=str, default='TopoMIL', choices=['TopoMIL'])

    ''' about PMSA (MAB - Transformer) '''
    parser.add_argument("-n_heads",    type=int, default=1)
    parser.add_argument('-norm',       action='store_true', help='use layer normalization')
    
    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=2, help='auxiliary batch size')
    parser.add_argument('-max_epoch', type=int, default=200, help='max epoch to run (cm16)')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate (cm16)')
    parser.add_argument('-wd', type=float, default=0.003, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[100], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-use_adam', action='store_true', help='optimizer choice')

    
    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='mil_set', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
                        default=arg_mode == 'test')  # train: enable logging / test: disable logging
    args = parser.parse_args()
    
    return args

import torch
import torch.nn as nn
from thop import profile, clever_format

def count_params(model, trainable_only=False, include_buffers=False):
    if trainable_only:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n_params = sum(p.numel() for p in model.parameters())
    if include_buffers:
        n_params += sum(b.numel() for b in model.buffers())
    return n_params

def measure_model(net, device=None, mode='inference', use_mil_input=False,
                  image_shape=(1, 3, 224, 224), mil_shape=(2, 4096, 512)):
    """
    测量模型复杂度，不改变原模型状态
    
    Args:
        net: 要测量的模型
        device: 计算设备
        mode: 'inference' 或 'training'
        use_mil_input: 是否使用MIL输入格式
        image_shape: 图像输入形状
        mil_shape: MIL输入形状
    
    Returns:
        dict: 包含参数量、FLOPs、内存使用等信息的字典
    """
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    original_training_state = net.training
    original_device = next(net.parameters()).device
    
    try:
        import copy
        net_for_prof = copy.deepcopy(net)
        
        if isinstance(net_for_prof, nn.DataParallel):
            net_for_prof = net_for_prof.module
        
        net_for_prof = net_for_prof.to(device)
        
        net_for_prof.eval()

        if use_mil_input:
            B, N, D = mil_shape
            dummy = torch.randn(B, N, D, device=device)
            thop_inputs = (dummy,)
        else:
            dummy = torch.randn(*image_shape, device=device)
            thop_inputs = (dummy,)

        mem = {}
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = net_for_prof(*thop_inputs)
                
            torch.cuda.synchronize()
            mem = {
                'max_allocated_GB': torch.cuda.max_memory_allocated() / 1024**3,
            }

        net_for_thop = copy.deepcopy(net_for_prof)
        macs, thop_params = profile(net_for_thop, inputs=thop_inputs, verbose=False)
        gflops = 2.0 * macs / 1e9  

        total_params = count_params(net_for_prof, trainable_only=False)
        train_params = count_params(net_for_prof, trainable_only=True)

        return {
            'params_total_M': total_params / 1e6,
            'params_trainable_M': train_params / 1e6,
            'THOP_params_M': thop_params / 1e6,
            'FLOPs_G(=2*MACs)': gflops,
            'mem': mem,
        }
        
    finally:
        net.train(original_training_state)
        
        if str(original_device) != str(next(net.parameters()).device):
            net = net.to(original_device)


