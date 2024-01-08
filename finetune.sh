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



for seed in 0 1 3; do for l in 4e-3; do for t in "coral_chorus_30min_few_shot"; do for f in True False; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 main_finetune_esc.py --log_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_australia --output_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_australia --model vit_base_patch16 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 3 --num_workers 2 --save_weight False --mixup 0.0 --seed ${seed} --finetune /mnt/data/tungtran/AudioMAE/logs/output_dir_australia/checkpoint-32.pth --freeze_base_model ${f}; done; done; done; done;


for seed in 0 1 3; do for l in 4e-3; do for f in True False; do for t in "indo_health_30min_few_shot" "indo_health_30min_trainBo_few_shot" "indo_location_30min_few_shot"; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29503 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_chorus_pretrained --output_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_chorus_pretrained --model vit_base_patch16 --dataset crs_indo --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --finetune /mnt/data/tungtran/AudioMAE/logs/output_dir_coral_chorus_mask0.8/checkpoint-32.pth --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 3 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed}; done; CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29503 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/coral_chorus_30min_few_shot_lr${l}_freeze${f}_seed${seed}_indo_pretrained --output_dir logs/coral_chorus_30min_few_shot_lr${l}_freeze${f}_seed${seed}_indo_pretrained  --model vit_base_patch16 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/class_labels_indices.csv --finetune /mnt/data/tungtran/AudioMAE/logs/output_dir_indo_mask0.8/checkpoint-32.pth --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 3 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed} ;done; done; done; 

for seed in 0 1 3; do for l in 4e-3; do for f in True False; do for t in "indo_health_30min_few_shot" "indo_health_30min_trainBo_few_shot" "indo_location_30min_few_shot"; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29506 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_audioset_pretrained --output_dir logs/${t}_lr${l}_freeze${f}_seed${seed}_audioset_pretrained --model vit_base_patch16 --dataset crs_indo --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_${t}/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 3 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed} --finetune /mnt/data/tungtran/AudioMAE/logs/AS_2M/pretrained.pth; done; CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29506 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/coral_chorus_30min_few_shot_lr${l}_freeze${f}_seed${seed}_audioset_pretrained --output_dir logs/coral_chorus_30min_few_shot_lr${l}_freeze${f}_seed${seed}_audioset_pretrained  --model vit_base_patch16 --dataset crs_coral_chorus --data_train /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/train.json --data_eval /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/val.json --label_csv /mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 3 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed}  --finetune /mnt/data/tungtran/AudioMAE/logs/AS_2M/pretrained.pth;done; done; done

# WATKINS:
for seed in 3; do for l in 4e-3; do for f in False; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29504 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/watkins_lr${l}_freeze${f}_seed${seed}_audioset --output_dir logs/watkins_lr${l}_freeze${f}_seed${seed}_audioset  --model vit_base_patch16 --dataset watkins --data_train dataset/watkins/train.json --data_eval dataset/watkins/val.json --label_csv dataset/watkins/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 31 --num_workers 2 --save_weight True --mixup 0.0 --freeze_base_model ${f} --seed ${seed} --audio_exp False --finetune logs/AS_2M/pretrained.pth; done; done; done; 


# DCASE
for seed in 0; do for l in 4e-5; do for f in False; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29505 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/dcase_lr${l}_freeze${f}_seed${seed}_australia --output_dir logs/dcase_lr${l}_freeze${f}_seed${seed}_australia  --model vit_base_patch16 --dataset dcase --data_train dataset/dcase/train.json --data_eval dataset/dcase/val.json --label_csv dataset/dcase/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 18 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed} --audio_exp False --finetune logs/output_dir_australia/checkpoint-32.pth --smoothing 0.0; done; done; done; 

# HICEAS:
for seed in 0; do for l in 4e-3 4e-4; do for f in False; do CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --master_port 29503 --nproc_per_node=1 main_finetune_esc.py --log_dir logs/hiceas_lr${l}_freeze${f}_seed${seed}_australia --output_dir logs/hiceas_lr${l}_freeze${f}_seed${seed}_australia  --model vit_base_patch16 --dataset hiceas --data_train dataset/hiceas/train.json --data_eval dataset/hiceas/val.json --label_csv dataset/hiceas/class_labels_indices.csv --epochs 60 --blr ${l} --batch_size 64 --warmup_epochs 4 --world_size 1 --nb_classes 1 --num_workers 2 --save_weight False --mixup 0.0 --freeze_base_model ${f} --seed ${seed} --audio_exp False --finetune logs/output_dir_australia/checkpoint-32.pth --smoothing 0.0; done; done; done; 




# EVAL ONLY:
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port 29506 main_finetune_esc.py --model vit_base_patch16 --dataset watkins --data_train dataset/watkins/test.json --data_eval dataset/watkins/test.json --label_csv dataset/watkins/class_labels_indices.csv --nb_classes 31 --world_size 1 --num_workers 2 --eval --resume logs/watkins_lr4e-3_freezeFalse_seed0_audioset/checkpoint-36.pth


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


for s in 0 1 3; do for i in {0..35}; do rm watkins_lr4e-3_freezeFalse_seed${s}_audioset/checkpoint-${i}.pth; done; done
