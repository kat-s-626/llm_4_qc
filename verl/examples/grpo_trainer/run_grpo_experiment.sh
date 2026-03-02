#!/bin/bash
#SBATCH --job-name=grover_grpo
#SBATCH --time=02:00:00
#SBATCH --mem=500GB
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=72
#SBATCH --nodes=16
#SBATCH --mail-type=ALL
#SBATCH --output="../sbatch_log/grover_grpo/%A.out"
set -x

# ================================
# Environment Setup
# ================================

export PROJECT_PATH="$HOME/rl_experiment"
export DATA_PATH="$HOME/data"

export HF_HOME="$HOME/.cache/HuggingFace"
export PIP_CACHE_DIR="$HOME/.cache/pip"

export FLASH_ATTENTION_FORCE_DISABLED=1 

verl_workdir=$PROJECT_PATH/verl                   # VERL installation directory
venv_path=$PROJECT_PATH/verl/venv                 # Virtual environment path
train_files=$DATA_PATH/grover_gates_20_50_subset_py/train.parquet        # Training dataset
val_files=$DATA_PATH/grover_gates_20_50_subset_py/test.parquet           # Validation dataset

# ================================
# Batch Size Configuration
# ================================

# Primary configuration - ONLY modify these values
PER_DEVICE_BATCH_SIZE=1                       # Samples per device for data collection
ROLLOUT_N=5                                       # Number of rollouts

# Hardware configuration (from SLURM)
NODES=${SLURM_NNODES:-2}                         # Number of nodes
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}          # GPUs per node
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))            # Total GPUs

# Calculate train batch size from per-device batch size
DATA_TRAIN_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NODES * GPUS_PER_NODE))


# Calculate PPO mini batch size
PPO_UPDATES_PER_ITERATION=4                      # Number of PPO updates per training iteration
MINI_BATCH_SIZE=$((DATA_TRAIN_BATCH_SIZE / PPO_UPDATES_PER_ITERATION))

# Validation: Check if mini batch size works with normalization
NORMALIZED_MINI_BATCH=$(( (MINI_BATCH_SIZE * ROLLOUT_N) / TOTAL_GPUS ))

# Apply tuning guide: Set micro batch size per GPU equal to normalized mini batch size
MICRO_BATCH_SIZE_PER_GPU=$NORMALIZED_MINI_BATCH

# Forward-only operations can use larger batch sizes (2x)
FORWARD_ONLY_MICRO_BATCH_SIZE=$((NORMALIZED_MINI_BATCH * 2))

MICRO_BATCHES_PER_MINI=$((NORMALIZED_MINI_BATCH / MICRO_BATCH_SIZE_PER_GPU))

# ================================
# Model and Training Setup
# ================================

# Model configuration
MODEL_PATH=$PROJECT_PATH/verl/merged_models/sft_grpo_grover/qwen3_8b
           # HuggingFace model path

# LoRA configuration
LORA_RANK=16                                     # LoRA rank
LORA_ALPHA=32                                    # LoRA alpha

# Sequence length configuration
MAX_PROMPT_LEN=800                              # Maximum prompt length
MAX_RESPONSE_LEN=13000                           # Maximum response length
MAX_TOKEN_LEN_PER_GPU=$((2 * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))  # Max tokens per GPU

# vLLM configuration
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))  # Conservative vLLM setting
MAX_NUM_BATCHED_TOKENS=$((MAX_MODEL_LEN * 6))   # Max batched tokens for vLLM
GPU_MEMORY_UTILIZATION=0.8                       # GPU memory utilization for vLLM
TENSOR_MODEL_PARALLEL_SIZE=1                     # Tensor parallelism size
ENABLE_CHUNKED_PREFILL=False                     # Enable chunked prefill

# Training hyperparameters
LEARNING_RATE=1e-6                               # Actor optimizer learning rate
USE_KL_LOSS=True                                 # Use KL divergence loss
KL_LOSS_COEF=0.001                               # KL loss coefficient
KL_LOSS_TYPE="low_var_kl"                        # KL loss type
ENTROPY_COEFF=0                                  # Entropy coefficient
USE_KL_IN_REWARD=False                           # Use KL in reward calculation

# Data configuration
FILTER_OVERLONG_PROMPTS=True                     # Filter prompts exceeding max length
DATA_TRUNCATION="error"                          # Truncation strategy

# Model optimization
USE_REMOVE_PADDING=True                          # Remove padding for efficiency
ENABLE_GRADIENT_CHECKPOINTING=True               # Enable gradient checkpointing
USE_DYNAMIC_BSZ=True                             # Use dynamic batch size

# FSDP configuration
ACTOR_PARAM_OFFLOAD=True                         # Offload actor parameters to CPU
ACTOR_OPTIMIZER_OFFLOAD=True                     # Offload actor optimizer to CPU
REF_PARAM_OFFLOAD=True                           # Offload reference parameters to CPU

# Trainer configuration
ALGORITHM_ADV_ESTIMATOR="grpo"                   # Advantage estimator (grpo/gae)
CRITIC_WARMUP=0                                  # Critic warmup epochs
LOGGER='["console"]'                             # Logger type
SAVE_FREQ=1                                      # Model save frequency (epochs)
TEST_FREQ=50                                     # Test frequency (epochs)
RESUME_MODE="auto"                               # Resume mode (auto/force/never)
TOTAL_EPOCHS=4                                   # Total training epochs

# Reward function
CUSTOM_REWARD_FUNCTION="verl/utils/reward_score/state_pred_reasoning.py"  # Custom reward function path

# Rollout configuration
ROLLOUT_NAME="vllm"                              # Rollout backend (vllm/sglang)

# Project naming
PROJECT_NAME="sft_grpo_qwen3_${MODEL_SIZE}_grover_gates_${MAX_RESPONSE_LEN}"
EXPERIMENT_NAME="sft_grpo_qwen3_${MODEL_SIZE}_grover_gates_subset_20"

# Automatic validation
if [ $((NORMALIZED_MINI_BATCH % MICRO_BATCH_SIZE_PER_GPU)) -ne 0 ]; then
    echo "ERROR: Batch configuration invalid."
    echo "  Normalized mini batch: $NORMALIZED_MINI_BATCH"
    echo "  Micro batch per GPU: $MICRO_BATCH_SIZE_PER_GPU"
    echo "  Try adjusting PER_DEVICE_BATCH_SIZE"
    exit 1
fi

echo "=== Optimized Batch Configuration (Tuning Guide Applied) ==="
echo "INPUT:"
echo "  - Per device batch size: $PER_DEVICE_BATCH_SIZE"
echo "CALCULATED:"
echo "  - Data train batch size: $DATA_TRAIN_BATCH_SIZE (global)"
echo "  - PPO mini batch size: $MINI_BATCH_SIZE (global)"
echo "  - Rollout N: $ROLLOUT_N"
echo "DERIVED:"
echo "  - Total GPUs: $TOTAL_GPUS"
echo "  - Normalized mini batch: $NORMALIZED_MINI_BATCH"
echo "  - Actor micro batch per GPU: $MICRO_BATCH_SIZE_PER_GPU (= normalized)"
echo "  - Forward-only micro batch: $FORWARD_ONLY_MICRO_BATCH_SIZE (2x for efficiency)"
echo "  - Micro batches per mini: $MICRO_BATCHES_PER_MINI"
echo "  - PPO updates per iteration: $PPO_UPDATES_PER_ITERATION"
echo "============================================="

# Activate virtual environment
source $venv_path/bin/activate

# Export PYTHONPATH to ensure VERL modules are found
export PYTHONPATH=$verl_workdir:$PYTHONPATH

# ================================
# Ray Cluster Setup
# ================================

# Get list of allocated nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)  # Create proper array (no quotes)

# First node becomes Ray head
head_node=${nodes_array[0]}

# Get IP address of head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle IPv6/IPv4 address selection
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}  # Use second address (IPv4)
    else
        head_node_ip=${ADDR[0]}  # Use first address
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379                    # Default Ray port
ip_head=$head_node_ip:$port  # Complete address
export RAY_ADDRESS=$ip_head  # Export for Ray client
export ip_head               # Export for compatibility
echo "IP Head: $ip_head"

# Print debugging information
echo "Python path: $(which python)"
echo "Ray version: $(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'Ray not found')"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# ================================
# Start Ray Head Node
# ================================

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_NTASKS_PER_NODE} -w "$head_node" --exact \
    bash -c "
    source $venv_path/bin/activate && \
    export PYTHONPATH=$verl_workdir:\$PYTHONPATH && \
    export HF_HOME=$HF_HOME && \
    export FLASH_ATTENTION_FORCE_DISABLED=1 && \
    ray start --head \
        --node-ip-address='$head_node_ip' \
        --port=$port \
        --num-cpus=${SLURM_NTASKS_PER_NODE} \
        --num-gpus=${SLURM_GPUS_PER_NODE} \
        --block
    " &

# Wait for head node to start
sleep 10

# ================================
# Start Ray Worker Nodes
# ================================

worker_num=$((SLURM_JOB_NUM_NODES - 1))  # Calculate number of workers

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_NTASKS_PER_NODE} -w "$node_i" --exact \
        bash -c "
        source $venv_path/bin/activate && \
        export PYTHONPATH=$verl_workdir:\$PYTHONPATH && \
        export HF_HOME=$HF_HOME && \
        export FLASH_ATTENTION_FORCE_DISABLED=1 && \
        ray start \
            --address='$ip_head' \
            --num-cpus=${SLURM_NTASKS_PER_NODE} \
            --num-gpus=${SLURM_GPUS_PER_NODE} \
            --block
        " &
    sleep 5
done

# Wait for all nodes to connect
sleep 10


# Check Ray cluster status (optional but helpful)
echo "Checking Ray cluster status..."
ray status --address=$ip_head || echo "Ray status check failed (may be normal)"

# ================================
# Run Training
# ================================

echo "Starting training on head node: $head_node"

# Run the main training script on the head node
PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" --exact \
    bash -c "
    source $venv_path/bin/activate && \
    export PYTHONPATH=$verl_workdir:\$PYTHONPATH && \
    export HF_HOME=$HF_HOME && \
    export FLASH_ATTENTION_FORCE_DISABLED=1 && \
    export RAY_ADDRESS=$ip_head && \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$ALGORITHM_ADV_ESTIMATOR \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
        data.max_prompt_length=$MAX_PROMPT_LEN \
        data.max_response_length=$MAX_RESPONSE_LEN \
        data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
        data.truncation=$DATA_TRUNCATION \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.lora_rank=$LORA_RANK \
        actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
        actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
        actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
        actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
        actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
        actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$FORWARD_ONLY_MICRO_BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
        actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$FORWARD_ONLY_MICRO_BATCH_SIZE \
        custom_reward_function.path=$CUSTOM_REWARD_FUNCTION \
        algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
        trainer.critic_warmup=$CRITIC_WARMUP \
        trainer.logger=$LOGGER \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.resume_mode=$RESUME_MODE \
        trainer.total_epochs=$TOTAL_EPOCHS \
        trainer.total_steps=2000 $@
    " 2>&1 | tee $PROJECT_PATH/verl_demo_slurm_$(date +%Y%m%d_%H%M%S).log