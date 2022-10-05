#!/bin/bash
#nvidia-smi

#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate gfla

export NGF=32
export DATAROOT='/shared/rsaas/shaoyuc3/dior-flowstyle/data'
export NET_G=dior
export NAME=dior_style_from_scratch_test
export PRETRAINED_FLOWNET_PATH='/shared/rsaas/shaoyuc3/dior-flowstyle/checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_101.pth'
export GENERATE_OUTPUT_PATH='/shared/rsaas/shaoyuc3/dior-flowstyle/checkpoints/dior_style_from_scratch_test/val_out'

python train.py --model dior \
--name $NAME --dataroot $DATAROOT --crop_size '256, 192' \
--batch_size 8 --lr 1e-4 --init_type orthogonal \
--loss_coe_seg 0 --val_output_dir $GENERATE_OUTPUT_PATH \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 20002 --n_epochs_decay 0 --lr_update_unit 4000 \
--print_freq 20 --display_freq 50 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 --continue_train \
--frozen_flownet --flownet_path $PRETRAINED_FLOWNET_PATH \
--random_rate 1 --warmup --perturb

rm -rf checkpoints/$NAME/latest_net_D*

python train.py --model dior \
--name $NAME --dataroot $DATAROOT --crop_size '256, 192' \
--batch_size 8 --lr 1e-4 --init_type orthogonal \
--loss_coe_seg 0.1 --val_output_dir $GENERATE_OUTPUT_PATH \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 40002 --n_epochs_decay 0 --lr_update_unit 4000 \
--print_freq 200 --display_freq 5000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 --continue_train \
--frozen_flownet \
--random_rate 1 --perturb

rm -rf checkpoints/$NAME/iter_40000_net_D*

python train.py --model dior \
--name $NAME --dataroot $DATAROOT --crop_size '256, 192' \
--batch_size 8 --lr 1e-5 --init_type orthogonal \
--loss_coe_seg 0.1 --val_output_dir $GENERATE_OUTPUT_PATH \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 60002 --n_epochs_decay 15000 --lr_update_unit 1000 \
--print_freq 200 --display_freq 5000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 0 --continue_train \
--epoch iter_40000 --epoch_count 40001 \
--frozen_flownet \
--random_rate 1 --perturb

#iter 80k random 25 -> 80
#iter 120k lr 1e-4 -> 5e-5