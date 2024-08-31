#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 0:30:00
#SBATCH -N 1           
#SBATCH -p shared-redstone
#SBATCH -C gpu_count:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --array=1-5%1  # 100 jobs in the array, 1 running at a time

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

#srun -o tiny_test.out --ntasks=1 -N 1 torchrun --nproc_per_node 2 gpt2_train.py --output_dir /tmp/test-clm --num_train_epochs 100 --config_file config.json --per_device_train_batch_size 12 --mixed_precision --save_steps 2000 &

# srun -o original.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --act_fun gelu --output_dir original &

#srun -o relu.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --output_dir relu &

# srun -o clip.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --act_fun clip --output_dir clip &



# srun -o rmsnorm.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --norm_type rms --output_dir rmsnorm 
# srun -o nobias.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 train_gpt2.py --micro_batch_size 16 --mlp_no_bias --output_dir nobias 
# srun --ntasks=1 -N 1 bash -c "echo 'Running on node:' $(hostname); nvidia-smi; torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_no_bias --mlp_renormalize --output_dir=renormalize"



# srun -o renormalize.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_no_bias --mlp_renormalize --output_dir=renormalize" 

# srun -o original.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --output_dir=original" 

# srun -o nobias.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_no_bias --output_dir=nobias" 

# srun -o rmsnorm.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_norm_type=rms --output_dir=rmsnorm" 

# srun -o renormalize_noskip.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_no_skip --mlp_renormalize --output_dir=renormalize_noskip" 

srun -o noskip.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --mlp_no_skip --output_dir=noskip" 

