# ResNet strikes back: An improved training procedure in timm

The codes are originated from [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

### Requirements
```angular2html
torch==1.11.0
torchvision==0.11.0
timm==0.5.4
```

### Performance

| Architecture | # Params | FLOPs | Baseline |    + AugMask    |
| :---: | :---: | :---: | :---: |:---------------:|
| ResNet50 | 25.6 M | 4.1 G | 79.7 | **80.0 (+0.3)** |
| ResNet101 | 44.5 M | 7.9 G | 81.4 | **82.1 (+0.7)** |
| ResNet152 | 60.2 M | 11.6 G | 81.8 | **82.8 (+1.0)** |


### AugMask training commands
- Enviroment variables
  ```bash
  data_path=/your/path/to/imagenet
  save_path=/your/path/to/save
  # Use target model name
  model_name=resnet50
  model_name=resnet101
  model_name=resnet152
  ```

- Command
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 train.py \
    ${data_path} \
    --model ${model_name} \
    --output ${save_path} \
    --img-size 224 \
    --epochs 300 \
    --batch-size 256 \
    --opt lamb \
    --lr 5e-3 \
    --sched cosine \
    --weight-decay 0.02\
    --warmup-epochs 5 \
    --cooldown-epochs 0 \
    --smoothing 0.0 \
    --drop 0.0 \
    --drop-path 0.05 \
    --aug-repeats 3 \
    --aa rand-m7-mstd0.5 \
    --mixup 0.1 \
    --cutmix 1.0 \
    --reprob 0.0 \
    --color-jitter 0.0 \
    --crop-pct 0.95 \
    --bce-loss \
    --native-amp \
    --log-interval 400 \
    --augsub masking \
    --augsub-ratio 0.5
```