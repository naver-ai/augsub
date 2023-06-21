# Data-Efficient architectures and training for Image classification

The codes are originated from [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

### Requirements
```angular2html
torch==1.11.0
torchvision==0.11.0a0
timm==0.3.2
```

### Performances

| Architecture | # params | FLOPs  | 400 epochs  |    + AugMask    | 800 epochs |    + AugMask    |
|:------------:|:--------:|:------:|:-----------:|:---------------:|:----------:|:---------------:|
| ViT-S/16     | 22.0 M   | 4.6 G  | 80.4        | **81.1 (+0.7)** | 81.4      | **81.7 (+0.3)** |
| ViT-B/16     | 86.6 M   | 17.5 G | 83.5        | **84.1 (+0.6)** | 83.8      | **84.2 (+0.4)** |
| ViT-L/16     | 304.4 M  | 61.6 G | 84.5        | **85.2 (+0.7)** | 84.9      | **85.3 (+0.4)** |
| ViT-H/14     | 632.1 M  | 167.4 G| 85.1        | **85.7 (+0.6)** | 85.2      | **85.7 (+0.5)** |

### AugMask training commands
- Enviroment variables
    ```bash
    data_path=/your/path/to/imagenet
    save_path=/your/path/to/save
    ```

- ViT-S
  - 400 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model deit_small_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path} --batch-size 256 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 4e-3 --weight-decay 0.03 --input-size 224 --drop 0.0 --drop-path 0.0 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - 800 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model deit_small_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path} --batch-size 256 --epochs 800 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 4e-3 --weight-decay 0.05 --input-size 224 --drop 0.0 --drop-path 0.05 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
- ViT-B
  - 400 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model deit_base_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 256 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.03 --input-size 192 --drop 0.0 --drop-path 0.1 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - 800 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model deit_base_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 256 --epochs 800 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.05 --input-size 192 --drop 0.0 --drop-path 0.2 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - Finetune
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model deit_base_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --finetune ${save_path}/pretrain/checkpoint.pth --output_dir ${save_path}/finetune --batch-size 64 --epochs 20 --smoothing 0.1 --reprob 0.0 --opt adamw --lr 1e-5 --weight-decay 0.1 --input-size 224 --drop 0.0 --drop-path 0.2 --mixup 0.8 --cutmix 1.0 --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 --dist-eval
    ```
- ViT-L
  - 400 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_large_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 32 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.03 --input-size 192 --drop 0.0 --drop-path 0.4 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - 800 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_large_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 32 --epochs 800 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.05 --input-size 192 --drop 0.0 --drop-path 0.45 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - Finetune
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_large_patch16_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --finetune ${save_path}/pretrain/checkpoint.pth --output_dir ${save_path}/finetune --batch-size 8 --epochs 20 --smoothing 0.1 --reprob 0.0 --opt adamw --lr 1e-5 --weight-decay 0.1 --input-size 224 --drop 0.0 --drop-path 0.45 --mixup 0.8 --cutmix 1.0 --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 --dist-eval
    ```
- ViT-H
  - 400 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_huge_patch14_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 32 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.03 --input-size 160 --drop 0.0 --drop-path 0.5 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - 800 epochs
    ```bash 
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_huge_patch14_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 32 --epochs 800 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.05 --input-size 160 --drop 0.0 --drop-path 0.6 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
    ```
  - Finetune
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model deit_huge_patch14_LS --augsub masking --augsub-ratio 0.5 --data-path ${data_path} --finetune ${save_path}/pretrain/checkpoint.pth --output_dir ${save_path}/finetune --batch-size 8 --epochs 20 --smoothing 0.1 --reprob 0.0 --opt adamw --lr 1e-5 --weight-decay 0.1 --input-size 224 --drop 0.0 --drop-path 0.55 --mixup 0.8 --cutmix 1.0 --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 --dist-eval
    ```
