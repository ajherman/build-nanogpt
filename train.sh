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

# MLP sphere

# srun -o original.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --n_epochs 5 --micro_batch_size=16 --output_dir=original" 

# srun -o pre_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --n_epochs 5 --micro_batch_size=16 --output_dir=pre_norm_sphere"


# MLP layer

# srun -o pre_post_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm layer --micro_batch_size=16 --output_dir=pre_post_norm_layer"

# srun -o post_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm none --mlp_post_norm layer --micro_batch_size=16 --output_dir=post_norm_layer"

# srun -o skip_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm none --mlp_skip_norm layer --micro_batch_size=16 --output_dir=skip_norm_layer"

# srun -o pre_skip_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_skip_norm layer --micro_batch_size=16 --output_dir=pre_skip_norm_layer"

# srun -o pre_post_skip_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm layer --mlp_skip_norm layer --micro_batch_size=16 --output_dir=pre_post_skip_norm_layer"


# MLP mixed

# srun -o pre_norm_layer_post_skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm sphere --mlp_skip_norm sphere --micro_batch_size=16 --output_dir=pre_norm_layer_post_skip_norm_sphere"

# Attn only

# srun -o just_attn.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm none --mlp_skip_norm none --attn_pre_norm layer --attn_post_norm layer --attn_skip_norm layer --micro_batch_size=16 --output_dir=just_attn"

# srun -o just_attn_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm none --mlp_skip_norm none --attn_pre_norm sphere --attn_post_norm sphere --attn_skip_norm sphere --micro_batch_size=16 --output_dir=just_attn_sphere"

# Attn mods to normalize MLP

# srun -o attn_pre_post_skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm sphere --attn_skip_norm sphere --micro_batch_size=16 --output_dir=attn_pre_post_skip_norm_sphere"

# srun -o attn_pre_post_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm sphere --attn_skip_norm none --micro_batch_size=16 --output_dir=attn_pre_post_norm_sphere"

#srun -o all_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm layer --mlp_post_norm layer --mlp_skip_norm layer --attn_pre_norm layer --attn_post_norm layer --attn_skip_norm layer --micro_batch_size=16 --output_dir=all_layer"

# srun -o attn_skip_norm_layer_pre_post_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm sphere --attn_skip_norm layer --micro_batch_size=16 --output_dir=attn_skip_norm_layer_pre_post_norm_sphere"

# srun -o attn_pre_skip_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm none --attn_skip_norm sphere --micro_batch_size=16 --output_dir=attn_pre_skip_norm_sphere"

# srun -o attn_pre_post_skip_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm layer --attn_post_norm layer --attn_skip_norm layer --micro_batch_size=16 --output_dir=attn_pre_post_skip_norm_layer"

srun -o attn_pre_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm none --attn_skip_norm none --micro_batch_size=16 --output_dir=attn_pre_norm_sphere"

# srun -o attn_pre_skip_norm_sphere_post_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm sphere --attn_post_norm layer --attn_skip_norm sphere --micro_batch_size=16 --output_dir=attn_pre_skip_norm_sphere_post_norm_layer"

srun -o attn_pre_skip_norm_layer_post_norm_sphere.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm layer --attn_post_norm sphere --attn_skip_norm layer --micro_batch_size=16 --output_dir=attn_pre_skip_norm_layer_post_norm_sphere"

srun -o attn_pre_post_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm layer --attn_post_norm layer --attn_skip_norm none --micro_batch_size=16 --output_dir=attn_pre_post_norm_layer"

srun -o attn_pre_skip_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm layer --attn_post_norm none --attn_skip_norm layer --micro_batch_size=16 --output_dir=attn_pre_skip_norm_layer"

srun -o attn_pre_norm_layer.out --ntasks=1 -N 1 bash -c "torchrun --nproc_per_node=4 train_gpt2.py --mlp_pre_norm sphere --mlp_post_norm sphere --mlp_skip_norm sphere --attn_pre_norm layer --attn_post_norm none --attn_skip_norm none --micro_batch_size=16 --output_dir=attn_pre_norm_layer"