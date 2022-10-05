export OUTPUT_DIR='checkpoints_fs/PBAFN_stage1_fs/generate'
export NAME=PBAFN_stage1_fs
export LOAD_EP=latest
export NET_G=dior
export NET_E=adgan
export NGF=32
export DATAROOT=...
export PRETRAINED_FLOWNET_PATH='checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_101.pth'


# generate images
python generate_all_flow.py --model dior --dataroot $DATAROOT \
--name $NAME --epoch $LOAD_EP --eval_output_dir $OUTPUT_DIR \
--netE $NET_E --netG $NET_G --ngf $NGF \
--n_cpus 4 --gpu_ids 0  --batch_size 1 \
--random_rate 1 --init_type 'orthogonal' --crop_size '256, 192' \
--flownet_path $PRETRAINED_FLOWNET_PATH --frozen_flownet
