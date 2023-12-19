# Multi-GPU:

python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
--batch_size 32 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 4 \
--blr 2e-4 --weight_decay 0.0001 \
--dataset crs_coral_chorus \
--data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus/train.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus/class_labels_indices.csv \
--roll_mag_aug True \
--decoder_mode 1 \
--output_dir output_dir_coral_chorus \
--log_dir output_dir_coral_chorus \
--distributed True \
--world_size 2 \
--num_workers 8

python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
--batch_size 32 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 4 \
--blr 2e-4 --weight_decay 0.0001 \
--dataset crs_indo \
--data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo/train.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo/class_labels_indices.csv \
--roll_mag_aug True \
--decoder_mode 1 \
--output_dir output_dir_indo \
--log_dir output_dir_indo \
--distributed True \
--world_size 2 \
--num_workers 8

for m in 0.3 0.5 0.8; do python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py --batch_size 32 --norm_pix_loss True --model mae_vit_base_patch16 --mask_ratio $m --epochs 33 --warmup_epochs 3 --save_every_epoch 4 --blr 2e-4 --weight_decay 0.0001 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus/train.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus/class_labels_indices.csv --roll_mag_aug True --decoder_mode 1 --output_dir output_dir_coral_chorus_mask${m} --log_dir output_dir_coral_chorus_mask${m} --distributed True --world_size 2 --num_workers 8; python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py --batch_size 32 --norm_pix_loss True --model mae_vit_base_patch16 --mask_ratio $m --epochs 33 --warmup_epochs 3 --save_every_epoch 4 --blr 2e-4 --weight_decay 0.0001 --dataset crs_indo --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo/train.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo/class_labels_indices.csv --roll_mag_aug True --decoder_mode 1 --output_dir output_dir_indo_mask${m} --log_dir output_dir_indo_mask${m} --distributed True --world_size 2 --num_workers 8; done

# Single-GPU:
python main_pretrain.py \
--batch_size 8 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 4 \
--blr 2e-4 --weight_decay 0.0001 \
--dataset crs_60sec \
--data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_60sec/train.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_60sec/class_labels_indices.csv \
--roll_mag_aug True \
--decoder_mode 1 \
--output_dir output_dir_60sec \
--log_dir output_dir_60sec \
