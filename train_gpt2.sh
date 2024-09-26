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

srun -o no_skip.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type sphere --attn_norm_type sphere --mlp_renormalize sphere --mlp_no_skip --micro_batch_size=16 --output_dir=no_skip" 

srun -o layer_no_skip.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type layer --attn_norm_type sphere --mlp_renormalize layer --mlp_no_skip --micro_batch_size=16 --output_dir=layer_no_skip" 

srun -o original_test.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --output_dir=original_test" 

srun -o mlp_renormalize_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type sphere --attn_norm_type sphere --mlp_renormalize sphere --micro_batch_size=16 --output_dir=mlp_renormalize_sphere" 

srun -o mlp_renormalize.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type sphere --attn_norm_type sphere --mlp_renormalize layer --micro_batch_size=16 --output_dir=mlp_renormalize" 

srun -o no_warmup.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --warmup_steps=0 --micro_batch_size=16 --output_dir=no_warmup"

srun -o full_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type sphere --attn_norm_type sphere --micro_batch_size=16 --output_dir=full_sphere" 

srun -o original.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --micro_batch_size=16 --output_dir=original" 

srun -o sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_norm_type sphere --micro_batch_size=16 --output_dir=sphere" 

