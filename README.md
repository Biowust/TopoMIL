# TopoMIL
[Feature Re-calibration based Multiple Instance Learning for Whole Slide Image Classification]

**Abstract:** *Weakly supervised whole-slide image (WSI) classification is a fundamental task in computational pathology, where multiple instance learning (MIL) provides a natural formulation for learning from slide-level labels and unlabeled patches. However, most existing MIL methods represent a WSI as an unordered bag and directly perform aggregation or relational modeling on all instances. Such a formulation is vulnerable to extensive background tissues and weakly relevant patches, and it also limits the modeling of spatial organization and contextual dependencies among diagnostically informative regions. In this paper, we propose TopoMIL, a topology-aware MIL framework for WSI classification. TopoMIL first employs a gated hard-attention module to identify discriminative instances from noisy bags and suppress redundant regions. The selected instances are then reorganized into a regular latent grid, converting the unordered bag into a structured representation for subsequent context modeling. On this grid, stacked bidirectional state-space blocks encode long-range regional dependencies, while a dense interaction aggregation module jointly exploits a classification token and masked global patch features to produce slide-level predictions. We systematically evaluate TopoMIL on CAMELYON16, CRC, and classical MIL benchmark datasets. Experimental results demonstrate that TopoMIL achieves competitive or superior performance across these datasets, indicating the effectiveness of combining key-instance filtering, latent structural reconstruction, and contextual interaction modeling for weakly supervised WSI classification.*

## Enviroment Requirements
* Ubuntu 22.4
* Python 3.10
* [CUDA 12.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 2.2.0](https://pytorch.org)

* ## Conda environment installation
```bash
conda env create --name TopoMIL python=3.10
conda activate TopoMIL
```
* run `pip install -r requirements.txt`

* ## Getting started

* ### Code Structure
```bash
TopoMIL/
-- checkpoints/ : default model checkpoint save location (includes pre-trained weights).
-- common/ : common utilities and functions.
-- configs/ : configuration files for pre-processing.
-- datasets/ : split and library files for a given dataset.
-- models /: consists of network definitions
        - /dataloaders/: defines the data loaders
-- scripts/ : train/test scripts
-- wsi_tools/ : collection of pre-processing scripts.

train.py: the main training script that requires a config file
test.py : the main testing script ''.
```

### Datasets | Pre-computed Features
* Download pre-computed features for Camelyon16 [Link](https://zenodo.org/record/6682429#.YrQavVxBwYt)
  * Unzip the features and modify the DATA variable in scripts i.e., train/wsi_frmil.sh and test/wsi_frmil.sh to re-train from scratch. (see other options) \
  Run ```bash scripts/train/wsi_frmil.sh ```
  * Pre-trained weights are stored in checkpoints/cm16/ \
    Run ```bash scripts/test/wsi_frmil.sh ```

### Model Architecture 
  * See  `models/mil_ss.py`
## Train 
Run ```bash scripts/train/wsi_frmil.sh ```

## Test 
Run ```bash scripts/test/wsi_test.sh ```

## References
Our implementation builds upon several existing publicly available code.

* [Weakly Supervised Temporal Action Loc (AAAI)](https://github.com/Pilhyeon/WTAL-Uncertainty-Modeling)
* [SetTransformer (ICML)](https://github.com/juho-lee/set_transformer)
* [DSMIL (CVPR)](https://github.com/binli123/dsmil-wsi) 
* [TransMIL (NeurIPS)](https://github.com/szc19990412/TransMIL)
* [RENET (ICCV)](https://github.com/dahyun-kang/renet/tree/main/datasets)
* [FRMIL](https://github.com/PhilipChicco/FRMIL)
