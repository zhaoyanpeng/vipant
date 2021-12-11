#!/usr/bin/sh

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=2

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

# bash bash/run_bimodal_va.sh bimodal 3

# train: via bimodal (vision-audio) pre-training 
eval="bimodal_va"
model_name="test"
mtask="
model_name=$model_name worker=CVALP port=$port num_gpus=$ngpu mode=$mode num_proc=$num_proc eval=False verbose=True 
+model/image=vit_val +model/audio=vit_val +model/text=dummy +model/loss=ce +optimizer=standard +running/audio=default 
model.audio.pre_encoder.in_channels=3 model.audio.pre_encoder.stride=[16,24] 
optimizer.warmup=False running.audio.norms=[-4.93839311,5.75751113] 
running.epochs=1 running.batch_size=2 running.peep_rate=50 running.save_rate=100 running.eval_samples=100
"

# config
extra="$mtask "
 
python train.py +running=$run_type $extra
