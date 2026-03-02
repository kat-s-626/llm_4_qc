#!/bin/bash
#SBATCH --job-name=grover_set_sft
#SBATCH --time=08:00:00
#SBATCH --mem=500GB
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --nodes=4
#SBATCH --mail-type=ALL
#SBATCH --output="../sbatch_log/grover_set_sft/%A.out"

set -x

# ================================
# Environment Setup
# ================================
export PROJECT_PATH="$HOME/rl_experiment"
export DATA_PATH="$HOME/data"
export HF_HOME="$HOME/.cache/HuggingFace"
export PIP_CACHE_DIR="$HOME/.cache/pip"
export FLASH_ATTENTION_FORCE_DISABLED=1 

PROJECT_NAME="grover_set_sft"

verl_workdir=$PROJECT_PATH/verl
venv_path=$PROJECT_PATH/verl/venv
train_files=$DATA_PATH/$PROJECT_NAME/train.parquet
val_files=$DATA_PATH/$PROJECT_NAME/test.parquet

MODEL_PATH="$PROJECT_PATH/verl/Qwen/Qwen3-8B-special"
save_path=$PROJECT_PATH/verl/checkpoints/$PROJECT_NAME/qwen3_8b_sft

# Set up master node for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

MICRO_BATCH_SIZE_PER_GPU=8

# Debug info
echo "=== SLURM Debug Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "========================"

# Launch training with torchrun via srun
srun --ntasks=$SLURM_NNODES \
     --ntasks-per-node=1 \
    bash -c "
    # Set Python path and environment
    export PYTHONPATH=$verl_workdir:\$PYTHONPATH
    export HF_HOME=$HF_HOME
    export FLASH_ATTENTION_FORCE_DISABLED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    echo \"Node \$SLURM_NODEID starting torchrun...\"
    source $venv_path/bin/activate
    
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.max_length=17500 \
        data.truncation=left \
        optim.lr=1e-4 \
        data.prompt_dict_keys=[formatted_prompt] \
        data.response_dict_keys=[formatted_completion] \
        data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
        model.partial_pretrain=$MODEL_PATH \
        model.strategy=fsdp \
        model.fsdp_config.cpu_offload=True \
        model.fsdp_config.offload_params=True \
        trainer.default_local_dir=$save_path \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$PROJECT_NAME_${MODEL_PATH#*/}_17500 \
        trainer.logger=console \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        trainer.total_epochs=1 \
        trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE \
        trainer.nnodes=$SLURM_NNODES \
        ulysses_sequence_parallel_size=$SLURM_NNODES \
        use_remove_padding=true
"