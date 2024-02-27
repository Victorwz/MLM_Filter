#!/bin/bash
num_gpu=$6
gpu_start_id=$1

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

start=`date +"%s"`

# Check if the first input parameter is "none"
if [ -z "$2" ]; then
  task="all"
else
  task=$2
fi

echo "Task is set to $task"

# Set batch size based on GPU type
if [[ $gpu_name == *"V100"* ]]; then
  batch_size=1
  workers=4
  tpg=$5
  for ((i=0;i<${num_gpu};i++ ))
  do
    {   
        if [ $i -gt 0 ]  
        then  
        sleep 30
        fi
        CUDA_VISIBLE_DEVICES=$i python llava_scoring_datacomp_batch_inference.py --gpu-id $(expr $i + $gpu_start_id) --batch-size $batch_size --workers $workers --tars-per-gpu $tpg --task $task --model-path $3 --tar-file-path $4
    } & 
  done
elif [[ $gpu_name == *"A100"* ]]; then
  batch_size=8
  workers=32
  tpg=$5
  for ((i=0;i<${num_gpu};i++ ))
  do
    {   
        if [ $i -gt 0 ]  
        then  
        sleep 30 
        fi
        CUDA_VISIBLE_DEVICES=$i python llava_scoring_datacomp_batch_inference.py --gpu-id $(expr $i + $gpu_start_id) --batch-size $batch_size --workers $workers --tars-per-gpu $tpg --task $task --model-path $3 --tar-file-path $4
    } & 
  done
else
  # Please adjust following your GPU type to avoid OOM
  batch_size=1
  workers=4
  tpg=$5
  for ((i=0;i<${num_gpu};i++ ))
  do
    {   
        if [ $i -gt 0 ]  
        then  
        sleep 30
        fi
        CUDA_VISIBLE_DEVICES=$i python llava_scoring_datacomp_batch_inference.py --gpu-id $(expr $i + $gpu_start_id) --batch-size $batch_size --workers $workers --tars-per-gpu $tpg --task $task --model-path $3 --tar-file-path $4
    } & 
  done
fi


wait
end=`date +"%s"`
echo "time: " `expr $end - $start`