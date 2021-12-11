#!/usr/bin/sh

export OMP_NUM_THREADS=32

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

if [ $ngpu -eq 0 ]; then
    ngpu=$2
fi
extra=$3

echo ${@:4}

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

#nohup python -m torch.utils.bottleneck train.py \
python train.py port=$port num_gpus=$ngpu \
	+running=$run_type $extra "${@:4}"
#> profile.new 2>&1 &
