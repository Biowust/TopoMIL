import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import os
import glob
import sys
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

def _safe_load_lib(path):
    """
    Robustly load a serialized bag library saved as .pth.
    Returns the loaded object (Dict or Tensor).
    """
    try:
        return torch.load(path, map_location='cpu')
    except (UnicodeDecodeError, UnicodeError):
        pass
    # Fallbacks with pickle for legacy encodings
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception:
        # Last attempt using torch.load with pickle_module
        return torch.load(path, map_location='cpu', pickle_module=pickle)

class WSIBagDataset(Dataset):
    def __init__(self, root=None, mode='train', batch=None, classes=2):
        self.root  = root 
        self.split = mode
        self.n_cls = classes
        self.batch = batch
        
        assert os.path.isdir(self.root), f'{self.root} is not a directory.'
        
        self.libs, self.path_labels = self.process()
        
        self.pos_weight, self.count_dict = self.computeposweight()
        
        self.bag_mu, self.bag_max, self.bag_min = 0, 0, 0
        self.ndims = 0 # 将在 get_bag_sizes 中自动检测
        
        if self.batch:
            self.bag_mu, self.bag_max, self.bag_min = self.get_bag_sizes()
            print(f'Bag Stats: mu {self.bag_mu:.1f} | min {self.bag_min} | max {self.bag_max} | dim {self.ndims}\n')
        
    def process(self):
        """
        扫描文件夹，返回所有文件路径和对应的标签列表
        """
        files = []
        labels = []
        for cls_id in range(self.n_cls):
            path_pattern = os.path.join(self.root, self.split, str(cls_id), "*.pth")
            feat_libs = glob.glob(path_pattern)
            feat_libs.sort() # 确保顺序一致
            
            files.extend(feat_libs)
            labels.extend([int(cls_id)] * len(feat_libs))
            
        return files, labels

    def computeposweight(self):
        """
        根据 process 阶段收集的 labels 列表计算权重，速度极快且不会报错
        """
        count_dict = {x: 0 for x in range(self.n_cls)}
        for lbl in self.path_labels:
            count_dict[lbl] += 1
            
        pos_count = count_dict.get(1, 0)
        total = len(self.path_labels)
        
        # 边界情况处理
        if total == 0:
            print(f'[Warning] No samples found in split "{self.split}".')
            return torch.tensor(1.0), count_dict
            
        if pos_count == 0 or pos_count == total:
            # 只有一类样本，权重设为 1
            return torch.tensor(1.0), count_dict
            
        # 标准正样本权重计算: (Total - Pos) / Pos
        weight = (total - pos_count) / pos_count
        return torch.tensor(weight, dtype=torch.float), count_dict

    def get_bag_sizes(self):
        """
        扫描所有文件以获取 Bag 大小统计，同时自动检测特征维度
        """
        bag_sizes = []
        feature_dim = 0
        print(f"Scanning {len(self.libs)} files to compute bag sizes...")
        
        for i, path in enumerate(self.libs):
            try:
                data = torch.load(path, map_location='cpu')
                
                if isinstance(data, dict):
                    for k in ['feature', 'features', 'data', 'encoding']:
                        if k in data:
                            data = data[k]
                            break
                
                if len(data.shape) == 3: # [N, 1, C]
                    data = data.squeeze()
                elif len(data.shape) == 4: # [N, H, W, C]
                    if data.shape[1] > 1 or data.shape[2] > 1:
                        data = data.mean(dim=(1, 2)) # Pooling
                    else:
                        data = data.reshape(data.shape[0], -1) # Flatten
                
                bag_sizes.append(data.shape[0])
                
                if feature_dim == 0:
                    feature_dim = data.shape[-1]
                    
            except Exception as e:
                print(f"Failed to load {path} in get_bag_sizes: {e}")
        
        if len(bag_sizes) == 0:
            raise ValueError("FATAL ERROR: No valid feature files loaded! Check your data path.")
            
        self.ndims = feature_dim # 更新类属性
        return np.mean(bag_sizes), np.max(bag_sizes), np.min(bag_sizes)

    def __getitem__(self, index):
        path = self.libs[index]
        
        target = self.path_labels[index]
        
        data = _safe_load_lib(path)
        
        if isinstance(data, dict):
            if 'class_id' in data:
                target = data['class_id']
                
            found = False
            for k in ['feature', 'features', 'data', 'encoding']:
                if k in data:
                    data = data[k]
                    found = True
                    break
            if not found:
                 data = list(data.values())[0]

        if not torch.is_tensor(data):
            data = torch.from_numpy(np.asarray(data)).float()
            
        
        if len(data.shape) == 2:
            pass # [N, C] 完美
            
        elif len(data.shape) == 3:
            # [N, 1, C] -> [N, C]
            data = data.squeeze()
            
        elif len(data.shape) == 4:
            # [N, 7, 7, C] -> [N, C] (Mean Pooling)
            if data.shape[1] > 1 or data.shape[2] > 1:
                data = data.mean(dim=(1, 2))
            else:
                # [N, 1, 1, C] -> [N, C]
                data = data.reshape(data.shape[0], -1)
                
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        if self.batch:
            num_inst = data.shape[0]
            curr_dim = data.shape[1]
            
            if self.ndims == 0: self.ndims = curr_dim
            
            bag_feats = torch.zeros((self.bag_max, self.ndims)).float()
            
            limit = min(num_inst, self.bag_max)
            bag_feats[:limit, :] = data[:limit, :]
            
            return bag_feats, target, [path], limit
        else:
            return data.float(), torch.tensor([target]), [path]

    def __len__(self):
        return len(self.libs)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

    
class WSIFolders(Dataset):

    def __init__(self,
                 root=None,
                 split='val',
                 class_map={'normal': 0, 'tumor': 1},
                 nslides=-1):

        self.classmap = class_map
        self.nslides = nslides
        self.split = split
        self.root = root
        # SimCLR patch Loader
        np.random.seed(0)
        
        print('Preprocessing folders .... ')
        lib = self.preprocess()
        
        self.slidenames = lib['slides']
        self.slides     = lib['slides']
        self.targets    = lib['targets']
        self.grid       = []
        self.slideIDX = []
        self.slideLBL = []
        
        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write(
                'Opening Folders : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))
        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        size = 256
        color_jitter = transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)
        data_trans   = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),]
                        )
        self.transform = {
            'orig' : transforms.Compose([transforms.Resize(size),transforms.ToTensor()]),
            'aug'  : data_trans,
        }
            

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        data = torch.load(path)
        
        if isinstance(data, dict):
            for k in ['feature', 'features', 'data', 'encoding']:
                if k in data:
                    data = data[k]
                    break

        if len(data.shape) == 2:
            pass 
            
        elif len(data.shape) == 3:
            data = data.squeeze() 
            
        elif len(data.shape) == 4:
            if data.shape[1] > 1 or data.shape[2] > 1:
                data = data.mean(dim=(1, 2))
            else:
                data = data.reshape(data.shape[0], -1)

        if len(data.shape) == 1:
            data = data.unsqueeze(0) 
            
        return data, label, path

    def __len__(self):
        return len(self.grid)

    def preprocess(self):
        grid = []
        targets = []
        slides = []
        class_names = [str(x) for x in range(len(self.classmap))]
        for i, cls_id in enumerate(class_names):
            slide_dicts = os.listdir(
                os.path.join(self.root, self.split, cls_id))
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):
                slide_folder = os.path.join(
                    self.root, self.split, cls_id, slide)
                if not os.path.isdir(slide_folder): continue
                
                grid_number = len(os.listdir(slide_folder))
                if grid_number == 0:
                    print("Skipped : ", slide, cls_id, ' | ', grid_number)
                    continue

                grid_p = []
                for id_patch in os.listdir(slide_folder):
                    grid_p.append(id_patch)

                if not slide_folder in slides:
                    slides.append(slide_folder)
                    grid.append(grid_p)
                    targets.append(int(cls_id))

        return {'slides': slides, 'grid': grid, 'targets': targets}