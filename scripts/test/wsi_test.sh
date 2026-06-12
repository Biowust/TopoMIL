# CM16
# ################
DATA="/mnt/data/ljc/data/cm16"
MDL=frmil
EXT="cm16_h8_simclr"
best_model_path="/mnt/data/ljc/outputs/frmil/cm16/2026_March_20_14h_50m_mamba/max_acc.pth"

python /mnt/data/ljc/FRMIL-mamba/test.py -batch 64 -n_heads 8 -mag 8.48 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 200 -lr 0.0001 -model_name $MDL -no_wandb -use_adam  -best_model_path $best_model_path -output_dir /mnt/data/ljc/outputs/frmil/cm16/2026_March_20_14h_50m_mamba


# CUDA_LAUNCH_BLOCKING=1 N_TRIALS=20 TUNE_TIMEOUT=3600 python /data/ljc/source/FRMIL-main/train.py -batch 2 -n_heads 8 -mag 8.48 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 200 -lr 0.001 -model_ext ${EXT} -model_name $MDL -no_wandb -use_adam -output_dir /data/ljc/source/outputs/
# # # ################

## crc
# DATA="/data/ljc/source/datasets/crc_simclr"
# MDL=frmil
# EXT="cm16_h8_simclr"
# best_model_path="/data/ljc/source/outputs/frmil/crc/2025_October_21_18h_05m_mamba/max_acc.pth"
# python /data/ljc/source/FRMIL-mamba/test.py -batch 32 -n_heads 8 -mag 8.48 -dataset crc -gpu 0 -data_name crc -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 200 -lr 0.0041 -model_name $MDL -no_wandb -use_adam -output_dir /data/ljc/source/outputs/frmil/crc/2025_October_21_18h_05m_mamba -best_model_path $best_model_path