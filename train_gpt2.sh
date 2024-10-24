#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 2:00:00
#SBATCH -N 1           
#SBATCH -p shared-redstone
#SBATCH -C gpu_count:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --array=1-20%1  # 100 jobs in the array, 1 running at a time

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

srun -o original.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --output_dir=original" 

srun -o pre_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --micro_batch_size=16 --output_dir=pre_norm_sphere" 

srun -o pre_post_skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --micro_batch_size=16 --output_dir=pre_post_skip_norm_sphere"

srun -o pre_post_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --micro_batch_size=16 --output_dir=pre_post_norm_sphere"

srun -o post_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm none --mlp_post_norm sphere --micro_batch_size=16 --output_dir=post_norm_sphere"

srun -o skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm none --mlp_skip_norm sphere --micro_batch_size=16 --output_dir=skip_norm_sphere"

srun -o pre_skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_skip_norm sphere --micro_batch_size=16 --output_dir=pre_skip_norm_sphere"



