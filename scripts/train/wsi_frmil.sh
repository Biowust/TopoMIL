# CM16
# ################
DATA="/mnt/data/ljc/data/cm16"
MDL=TopoMIL
EXT="cm16_h8_simclr"

python /mnt/data/ljc/TopoMIL-mamba/train.py -batch 64 -n_heads 8 -mag 8.48 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 150 -seed 257 -lr 1e-05 -wd 1e-05 -model_name $MDL -no_wandb -use_adam -output_dir /mnt/data/ljc/outputs



# 单卡运行（当前配置）
# N_TRIALS=50 python /mnt/data/ljc/TopoMIL-mamba/train_optuna.py -batch 32 -n_heads 8 -mag 8.48 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 200 -lr 0.0001  -model_name $MDL -no_wandb -use_adam -output_dir /mnt/data/ljc/outputs


# crc
# ################
# EXT="crc"
# DATA="/mnt/data/ljc/data/crc_features"
# MDL=TopoMIL
# python /mnt/data/ljc/TopoMIL-mamba/train_crc.py -batch 16 -n_heads 8 -mag 8.48 -dataset crc -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -lr 0.005 -wd 0.001 -num_feats 768 -max_epoch 50 -gamma 0.7 -model_name $MDL -no_wandb -use_adam -output_dir /mnt/data/ljc/outputs

# N_TRIALS=50 python /mnt/data/ljc/TopoMIL-mamba/train_optuna_crc.py \
#   -batch 32 -n_heads 8 -mag 8.48 -dataset crc -gpu 0 -data_name crc \
#   -extra_dir "frmil_crc_simple" -data_dir /mnt/data/ljc/datasets/crc_simclr \
#   -max_epoch 150 -lr 0.0001 -model_name TopoMIL -no_wandb -use_adam \
#   -output_dir /mnt/data/ljc/outputs