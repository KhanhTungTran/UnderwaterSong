# TASK: classification of location of recordings
# FROM PRE-TRAINED:
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 main_finetune_esc.py \
--log_dir logs/location_finetune \
--output_dir location_finetune \
--model vit_base_patch16 \
--dataset crs \
--data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound/train.json \
--data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound/val.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound/class_labels_indices.csv \
--finetune /mnt/data/tungtran/AudioMAE/output_dir_32batchsize_2GPU/checkpoint-32.pth \
--epochs 10 \
--blr 1e-4 \
--batch_size 8 \
--warmup_epochs 4 \
--dist_eval \
--world_size 2 \
--nb_classes 6

# FROM SCRATCH:
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main_finetune_esc.py \
--log_dir logs/location_scratch \
--output_dir location_scratch \
--model vit_base_patch16 \
--dataset crs \
--data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound/train.json \
--data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound/val.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound/class_labels_indices.csv \
--epochs 10 \
--blr 1e-4 \
--batch_size 8 \
--warmup_epochs 4 \
--dist_eval \
--world_size 2 \
--nb_classes 6


for seed in 3 4; do for l in 1e-3 1e-4 1e-5; do for t in "coral_chorus_30min_few_shot"; do for m in 0.3 0.5 0.8; do for f in True False; do CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 main_finetune_esc.py --log_dir logs/${t}_lr${l}_mask${m}_freeze${f}_seed${seed}_finetune --model vit_base_patch16 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --finetune /mnt/data/tungtran/AudioMAE/logs/output_dir_indo_mask${m}/checkpoint-32.pth --epochs 10 --blr ${l} --batch_size 8 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 3 --num_workers 0 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed}; done; done; CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 main_finetune_esc.py --log_dir logs/${t}_lr${l}_seed${seed}_scratch --model vit_base_patch16 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --epochs 10 --blr ${l} --batch_size 8 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 3 --num_workers 0 --save_weight False --mixup 0.0 --seed ${seed}; done; done; done; 

for seed in 3 4; do for l in 1e-3 1e-4 1e-5; do for t in "indo_health_30min_few_shot" "indo_health_30min_trainBa_few_shot" "indo_health_30min_trainBo_few_shot" "indo_health_30min_trainSa_few_shot" "indo_location_30min_few_shot"; do for m in 0.3 0.5 0.8; do for f in True False; do CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29503 --nproc_per_node=2 main_finetune_esc.py --log_dir logs/${t}_lr${l}_mask${m}_freeze${f}_seed${seed}_finetune  --model vit_base_patch16 --dataset crs_indo --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --finetune /mnt/data/tungtran/AudioMAE/logs/output_dir_coral_chorus_mask${m}/checkpoint-32.pth --epochs 10 --blr ${l} --batch_size 8 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 3 --num_workers 0 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed}; done; done; CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29503 --nproc_per_node=2 main_finetune_esc.py --log_dir logs/${t}_lr${l}_seed${seed}_scratch  --model vit_base_patch16 --dataset crs_indo --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --epochs 10 --blr ${l} --batch_size 8 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 3 --num_workers 0 --save_weight False --mixup 0.0 --seed ${seed}; done; done; done



# Multilabel fish classification
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main_finetune_as.py \
--log_dir logs/fish_multilabel_finetune_60sec \
--output_dir fish_multilabel_finetune_60sec \
--model vit_base_patch16 \
--dataset fish_crs \
--data_train /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/tr_1.json \
--data_eval /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/te_1.json \
--label_csv /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/class_labels_indices_crs.csv \
--epochs 30 \
--blr 1e-4 \
--batch_size 4 \
--warmup_epochs 4 \
--dist_eval \
--world_size 2 \
--nb_classes 10 \
--num_workers 4 \
--weight_sampler False \
--weight_csv /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/tr_1_weight.csv \
--save_weight False \
--mixup 0.0 \
--finetune /mnt/data/tungtran/AudioMAE/output_dir_60sec/checkpoint-32.pth


for l in 1e-3; do for w in False; do for split in 1 2 3 4 5; do CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main_finetune_as.py --log_dir logs/fish_multilabel_finetune_60sec_lr${l}_weight${w}_nomixup_split${split} --output_dir fish_multilabel_finetune_60sec_lr${l}_weight${w}_nomixup_split${split} --model vit_base_patch16 --dataset fish_crs --data_train /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/tr_${split}.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/te_${split}.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/class_labels_indices_crs.csv --epochs 30 --blr ${l} --batch_size 4 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 10 --num_workers 4 --weight_sampler False --finetune /mnt/data/tungtran/AudioMAE/output_dir_60sec/checkpoint-32.pth --save_weight False --mixup 0.0; CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main_finetune_as.py --log_dir logs/fish_multilabel_scratch_60sec_lr${l}_weight${w}_nomixup_split${split} --output_dir fish_multilabel_scratch_60sec_lr${l}_weight${w}_nomixup_split${split} --model vit_base_patch16 --dataset fish_crs --data_train /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/tr_${split}.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/te_${split}.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/fish_multilabel_metadata/class_labels_indices_crs.csv --epochs 30 --blr ${l} --batch_size 4 --warmup_epochs 4 --dist_eval --world_size 2 --nb_classes 10 --num_workers 4 --weight_sampler False --save_weight False --mixup 0.0; done; done; done; 
