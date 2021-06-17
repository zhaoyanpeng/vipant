## A running example
```
python train.py +model/image=vit +model/audio=vit +model/loss=ce +optimizer=standard +running=train
```
## A DDP running example
```
#!/usr/bin/sh

root=./

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES="1,3,4,6"

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

#nohup python -m torch.utils.bottleneck train.py \
python train_ddp.py port=$port num_gpus=$ngpu \
        +model/image=vit \
        +model/audio=vit \
        +model/loss=ce \
        +optimizer=standard \
        +running=$run_type
#> profile.new 2>&1 &
```
