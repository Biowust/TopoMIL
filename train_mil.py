import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import torch.nn.functional as F
import sys, argparse, os, copy, itertools
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from collections import OrderedDict
from common.utils import plot_cv_roc_pr_curves


try:
    from models.mil_ss import TopoMIL
except ImportError:
    print("Warning: Could not import TopoMIL. Please check your file path.")

def get_data(file_path):
    """Load MIL dataset from SVM format file"""
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]
    data_list = []    
    for i in range(0, df.shape[0]):  
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        idi = int(ids[0])
        idb = int(ids[1])
        idc = int(ids[2])
        data = data[1:]
        feature_vector = np.zeros(len(data))  
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([idi, idb, idc, feature_vector])
    return data_list

def get_bag(data, idb):
    """Extract bag data by bag ID"""
    data_array = np.array(data, dtype=object)
    bag_id = data_array[:, 1]
    return data_array[np.where(bag_id == idb)]

def epoch_train(bag_ins_list, optimizer, model, args, bce_weight, ce_weight):
    """Training epoch for TopoMIL-mamba model (Adaptive Return)"""
    epoch_loss = 0
    model.train()
    for i, data in enumerate(bag_ins_list):
        optimizer.zero_grad()
        data_bag_list = shuffle(data[1])
        data_tensor = torch.tensor(np.stack(data_bag_list)).float().cuda()
        data_tensor = data_tensor[:, 0:args.num_feats]
        
        # Padding
        if data_tensor.size(1) < args.num_feats:
            pad_size = args.num_feats - data_tensor.size(1)
            padding = torch.zeros(data_tensor.size(0), pad_size, device=data_tensor.device)
            data_tensor = torch.cat([data_tensor, padding], dim=1)
        
        # Add batch dimension
        data_tensor = data_tensor.unsqueeze(0)  # [1, N, D]
        label = torch.tensor(int(np.clip(data[0], 0, 1)), dtype=torch.long).cuda()
        
        outputs = model(data_tensor)
        
        logits = None
        max_c = None
        
        if isinstance(outputs, tuple) and len(outputs) == 3:
            logits, query, max_c = outputs
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            logits = outputs[0]
        else:
            logits = outputs
            
        loss_bag = F.cross_entropy(logits, label.unsqueeze(0), weight=ce_weight)
        
        if max_c is not None:
            max_prediction = torch.sigmoid(torch.max(max_c, 1)[0])
            loss_max = F.binary_cross_entropy(max_prediction, label.float().unsqueeze(0), weight=bce_weight)
            loss_total = 0.5*loss_bag + 0.5*loss_max
        else:
            loss_total = loss_bag
        
        loss_total.backward()
        optimizer.step()  
        epoch_loss = epoch_loss + loss_total.item()
        
    return epoch_loss / len(bag_ins_list)

def epoch_test(bag_ins_list, model, args, bce_weight, ce_weight):
    """Testing epoch for TopoMIL-mamba model (Adaptive Return)"""
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            bag_labels.append(np.clip(data[0], 0, 1))

            data_tensor = torch.tensor(np.stack(data[1])).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            
            # Padding
            if data_tensor.size(1) < args.num_feats:
                pad_size = args.num_feats - data_tensor.size(1)
                padding = torch.zeros(data_tensor.size(0), pad_size, device=data_tensor.device)
                data_tensor = torch.cat([data_tensor, padding], dim=1)
            
            data_tensor = data_tensor.unsqueeze(0)  # [1, N, D]
            label = torch.tensor(int(np.clip(data[0], 0, 1)), dtype=torch.long).cuda()
            
            outputs = model(data_tensor)
            
            logits = None
            max_c = None
            
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits, query, max_c = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                logits = outputs[0]
            else:
                logits = outputs
            
            loss_bag = F.cross_entropy(logits, label.unsqueeze(0), weight=ce_weight)
            
            if max_c is not None:
                max_prediction_inst = torch.sigmoid(torch.max(max_c, 1)[0])
                loss_max = F.binary_cross_entropy(max_prediction_inst, label.float().unsqueeze(0), weight=bce_weight)
                loss_total = 0.5*loss_bag + 0.5*loss_max
            else:
                loss_total = loss_bag
            
            bag_prob = torch.softmax(logits, dim=1)[:, 1]
            
            bag_predictions.append(bag_prob.cpu().item())
            epoch_loss = epoch_loss + loss_total.item()
            
    epoch_loss = epoch_loss / len(bag_ins_list)
    return epoch_loss, bag_labels, bag_predictions

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary', zero_division=0)
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore

def cross_validation_set(in_list, fold, index):
    csv_list = copy.deepcopy(in_list)
    n = int(len(csv_list)/fold)
    chunked = [csv_list[i:i+n] for i in range(0, len(csv_list), n)]
    if len(chunked) > fold:
        chunked[-2].extend(chunked[-1])
        chunked.pop()
        
    test_list = chunked[index]
    train_chunks = chunked[:index] + chunked[index+1:]
    train_list = list(itertools.chain.from_iterable(train_chunks))
    return train_list, test_list

def compute_pos_weight(bags_list):
    pos_count = 0
    for item in bags_list:
        pos_count = pos_count + np.clip(item[0], 0, 1)
    if pos_count == 0: return 1.0
    return (len(bags_list)-pos_count)/pos_count

def main():
    parser = argparse.ArgumentParser(description='Train TopoMIL-mamba on classical MIL datasets')
    parser.add_argument('--datasets', default='musk1', type=str, help='musk1, musk2, elephant, fox, tiger')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=40, type=int)
    parser.add_argument('--cv_fold', default=10, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--model', default='TopoMIL', type=str)
    parser.add_argument('--num_feats', default=166, type=int)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--mag', default=8.48, type=float)
    args = parser.parse_args()
   
    base_path = '/mnt/data/ljc/data/mil_dataset'
    if args.datasets == 'musk1':
        data_path = f'{base_path}/Musk/musk1norm.svm'
        args.num_feats = 166
    elif args.datasets == 'musk2':
        data_path = f'{base_path}/Musk/musk2norm.svm'
        args.num_feats = 166
    elif args.datasets == 'elephant':
        data_path = f'{base_path}/Elephant/data_100x100.svm'
        args.num_feats = 230
    elif args.datasets == 'fox':
        data_path = f'{base_path}/Fox/data_100x100.svm'
        args.num_feats = 230
    elif args.datasets == 'tiger':
        data_path = f'{base_path}/Tiger/data_100x100.svm'
        args.num_feats = 230  
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    data_all = get_data(data_path)
    
    # Prepare bag data
    bag_ins_list = []
    bag_ids = [row[1] for row in data_all]
    num_bag = max(bag_ids) + 1
    
    for i in range(num_bag):
        bag_data = get_bag(data_all, i)
        if len(bag_data) == 0: continue
        bag_label = bag_data[0, 2]
        bag_vector = bag_data[:, 3]
        bag_ins_list.append([bag_label, bag_vector])
    
    bag_ins_list = shuffle(bag_ins_list)
    
    valid_bags = 1
    attempt = 0
    while(valid_bags):
        attempt += 1
        bag_ins_list = shuffle(bag_ins_list)
        _, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=0)
        labels = [x[0] for x in test_list]
        if 1 in labels and 0 in labels:
            valid_bags = 0 
        if attempt > 20: break

    acs = []
    print('Dataset: ' + args.datasets)
    for k in range(0, args.cv_fold):
        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        bags_list, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=k)

        args.data_name = args.datasets
        args.num_class = 2
        
        if args.num_feats % args.n_heads != 0:
            args.num_feats = ((args.num_feats + args.n_heads - 1) // args.n_heads) * args.n_heads
        
        model = TopoMIL(args).cuda()
        
        pos_weight = torch.tensor(compute_pos_weight(bags_list))
        bce_weight = pos_weight.cuda()
        ce_weight = torch.tensor([1.0, pos_weight.item()]).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0)
        
        optimal_ac = 0
        for epoch in range(0, args.num_epoch):
            train_loss = epoch_train(bags_list, optimizer, model, args, bce_weight, ce_weight)
            test_loss, bag_labels, bag_predictions = epoch_test(test_list, model, args, bce_weight, ce_weight)
            
            accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_predictions)
            
            sys.stdout.write('\r Epoch [%d/%d] train loss: %.4f, test loss: %.4f, acc: %.4f, auc: %.4f, f1: %.4f ' % 
                  (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, fscore))
            
            if accuracy > optimal_ac:
                optimal_ac = accuracy
                
            scheduler.step()
        print('\n Optimal accuracy: %.4f ' % (optimal_ac))
        acs.append(optimal_ac)
        
    print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))

if __name__ == '__main__':
    main()