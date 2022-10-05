export FLOWNET_PATH=...

python -m torch.distributed.launch --nproc_per_node=1 --master_port=7130 train_flow.py --name PBAFN_stage1_fs \
--verbose --tf_log --batch_size 4 --launcher pytorch --checkpoints_dir checkpoints_fs \
--pool_size 0 --display_freq 100 --save_latest_freq 1000 --save_epoch_freq 20 \
--lr 0.00001 --netG global --num_gpus 1 --model dior --continue_train \
--flownet_path $FLOWNET_PATH --continue_train
