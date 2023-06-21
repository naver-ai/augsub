## MAE finetuning

The codes are originated from [https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)


### Requirements
```angular2html
torch==1.11.0
torchvision==0.11.0
timm==0.3.2
```

### Performance

| Architecture | Finetuning Epochs | Baseline |    + AugMask    |
|:------------:|:-----------------:|:--------:|:---------------:|
| ViT-B/16     | 100               | 83.6     | **83.9 (+0.3)** |
| ViT-L/16     | 50                | 85.9     | **86.1 (+0.2)** |
| ViT-H/14     | 50                | 86.9     | **87.2 (+0.3)** |

### AugMask finetuning commands

Finetuning requires MAE pretrained weights. Please download MAE weights from [original repository](https://github.com/facebookresearch/mae)

- Enviroment variables
  ```bash
  data_path=/your/path/to/imagenet
  save_path=/your/path/to/save
  weight_path=/your/path/to/mae_pretrain_vit_base.pth
  ```
- ViT-B
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_finetune.py \
      --model vit_base_patch16 \
      --data_path ${data_path} \
      --finetune ${weight_path} \
      --output_dir ${save_path} \
      --batch_size 128 \
      --accum_iter 1 \
      --epochs 100 \
      --blr 5e-4 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
      --augsub masking --augsub_ratio 0.5 \
      --dist_eval 
  ```

- ViT-L
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_finetune.py \
      --model vit_large_patch16 \
      --data_path ${data_path} \
      --finetune ${weight_path} \
      --output_dir ${save_path} \
      --batch_size 32 \
      --accum_iter 4 \
      --epochs 50 \
      --blr 1e-3 --layer_decay 0.75 \
      --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
      --augsub masking --augsub_ratio 0.5 \
      --dist_eval 
  ```

- ViT-H
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main_finetune.py \
      --model vit_huge_patch16 \
      --data_path ${data_path} \
      --finetune ${weight_path} \
      --output_dir ${save_path} \
      --batch_size 16 \
      --accum_iter 1 \
      --epochs 50 \
      --blr 1e-3 --layer_decay 0.75 \
      --weight_decay 0.05 --drop_path 0.3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
      --augsub masking --augsub_ratio 0.5 \
      --dist_eval 
  ```
