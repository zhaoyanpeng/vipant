#!/usr/bin/sh

root=./

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=8

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

# bash bash/run_bimodal_at.sh trimodal 3
# train: finetune via the audio-text task (AudioCaps) 
eval="bimodal_at"
model_file="bimodal_16x24_00071478.pth" # VA-pre-trained model
model_file="notafile"
model_name="test"
mtask="
model_name=$model_name monitor=VALMonitor worker=CVALP port=$port num_gpus=$ngpu mode=$mode num_proc=$num_proc eval=False verbose=True
+model/image=vit_val +model/audio=vit_val +model/text=transformer_val +model/loss=ce_val +optimizer=standard +running/audio=default 
model.audio.pre_encoder.in_channels=3 model.audio.pre_encoder.stride=[16,24] 
optimizer.warmup=False running.audio.norms=[-4.93839311,5.75751113]
running.siamese.alive=True running.imagine=False model.loss.va=False 
running.batch_size=64 running.peep_rate=1 running.prompt=\"\" 

model_file=$model_file
+running.rnd_cap=True

running.data_name=audiocaps_train running.eval_name=audiocaps_val running.test_name=audiocaps_test 
running.eval_samples=250 running.test_samples=250 running.train_samples=0.1 
"

# config
extra="$mtask"

python train.py +running=$run_type $extra 
