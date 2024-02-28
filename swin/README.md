# Swin Transformer

The codes are originated from [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)


### Requirements
```angular2html
torch==1.11.0
torchvision==0.11.0
timm==0.3.2
```

### Performance

| Architecture | # Params | FLOPs | Baseline |    + MaskSub    |
| :---: | :---: | :---: | :---: |:---------------:|
| Swin-T | 28.3 M | 4.5 G | 81.3 | **81.4 (+0.1)** |
| Swin-S | 49.6 M | 8.7 G | 83.0 | **83.4 (+0.4)** |
| Swin-B | 87.9 M | 15.4 G | 83.5 | **83.9 (+0.4)** |

### MaskSub training commands
- Enviroment variables
  ```bash
  data_path=/your/path/to/imagenet
  save_path=/your/path/to/save
  # Use config file of target model
  config_file=swin_tiny_patch4_window7_224.yaml
  config_file=swin_small_patch4_window7_224.yaml
  config_file=swin_base_patch4_window7_224.yaml
  ```

- Command
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main.py \
     --cfg configs/swin/${config_file} --data-path ${data_path} --output ${save_path} --batch-size 128 \
     --augsub masking --augsub-ratio 0.5 
    ```
