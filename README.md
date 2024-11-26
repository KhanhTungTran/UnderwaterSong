# UnderwaterSong


This repo hosts the code and models of **UnderwaterSong**. It is based on the code of "[Masked Autoencoders that Listen](http://arxiv.org/abs/2207.06405)".

### 1. Installation
- This repo follows the [MAE repo](https://github.com/facebookresearch/mae), Installation and preparation follow that repo.
- Copy files and patch the timm package by ``bash timm_patch.sh'' (Please change the path to your own timm package path). We use timm==0.3.2, for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.
- Please find [mae_env.yml](./mae_env.yml) for all the dependencies.
- You may also use download the conda-packed [conda env](https://drive.google.com/file/d/1ECVmVyscVqmhI7OQa0nghIsWVaZhZx3q/view?usp=sharing), untar it, and then:
```
source path_to_env/bin/activate
```

### 2. Prepare data:
For the sound detection task, have a look at [hiceas data](dataset/hiceas/train.json) for an example.
```
{
    "wav": path_to_audio_file,
    "length": lenght_of_audio_file_in_seconds,
    "annotations": [
    {
        "st": start_time_1,
        "ed": end_time_1,
        "label": sound_type_1
    },
    {
        "st": start_time_2,
        "ed": end_time_2,
        "label": sound_type_2
    }
    ]
},
```

### 3. Inference 
The command below can be run to evaluate performance on hiceas dataset
```
CUDA_VISIBLE_DEVICES="0" python main_finetune_esc.py --model vit_base_patch16 --dataset hiceas --data_train dataset/hiceas/test.json --data_eval dataset/hiceas/test.json --label_csv dataset/hiceas/class_labels_indices.csv --nb_classes 1 --world_size 1 --num_workers 2 --eval --resume path_to_checkpoint
```

### Checkpoints:
2. UnderwaterSong pretrained + [finetuned on hiceas (sound detection task)](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link)
