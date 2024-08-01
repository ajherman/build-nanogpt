#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1           
#SBATCH -p shared-redstone
#SBATCH -C gpu_count:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
module load miniconda3

# Enable Python fault handler
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' CUDA error checking
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

source activate /vast/home/ajherman/miniconda3/envs/transformer
#pip install datasets
#pip install tiktoken
#export PATH="/vast/home/ajherman/miniconda3/envs/transformer/bin:$PATH"

# srun -o original.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir original --num_train_epochs 2 --config_file medium.json --per_device_train_batch_size 13 --mixed_precision --save_steps 200 --logging_steps 50 --eval_steps 50 --gradient_accumulation_steps 10 --max_step 200000 --load_from_checkpoint &

#srun -o relu.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir relu --num_train_epochs 2 --config_file relu.json --per_device_train_batch_size 13 --mixed_precision --save_steps 200 --logging_steps 50 --eval_steps 50 --gradient_accumulation_steps 10 --max_steps 200000 &

#srun -o fast.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir fast --num_train_epochs 2 --config_file medium.json --per_device_train_batch_size 16 --mixed_precision --save_steps 200 --logging_steps 50 --eval_steps 50 --gradient_accumulation_steps 8 --max_step 200000 --learning_rate 1e-2 --load_from_checkpoint &

# srun -o pretrained.out --ntasks=1 -N 1 torchrun --nproc_per_node 2 gpt2_train.py --output_dir pretrained --num_train_epochs 2 --config_file medium.json --per_device_train_batch_size 13 --mixed_precision --logging_steps 50 --eval_steps 50 --max_step 1 --no_train --learning_rate 0.0 &

#srun -o test.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 test.py &

#nvidia-smi

#srun -o tiny_test.out --ntasks=1 -N 1 torchrun --nproc_per_node 2 gpt2_train.py --output_dir /tmp/test-clm --num_train_epochs 100 --config_file config.json --per_device_train_batch_size 12 --mixed_precision --save_steps 2000 &

srun -o original.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --act_fun gelu --output_dir original &

# srun -o relu.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --output_dir relu &

#srun -o clip.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --act_fun clip --output_dir clip &

