#!/bin/bash
#SBATCH --job-name=dna
#SBATCH --open-mode=append
#SBATCH --output=/tmp/%x_%A_%a.out
#SBATCH --error=/tmp/%x_%A_%a.error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=384G
#SBATCH --qos=
#SBATCH --account=
#SBATCH --time=168:00:00
#SBATCH --array=0-19

# Load modules and activate environment
source ~/.bashrc
conda activate dag

# Extract WORLD_SIZE from SBATCH ntasks-per-node setting
WORLD_SIZE=$(grep -oP '(?<=#SBATCH --ntasks-per-node=)\d+' "$0")

# ---- CPU/Thread Settings ----
CPUS_PER_TASK=$(grep -oP '(?<=#SBATCH --cpus-per-task=)\d+' "$0")
RESERVED_CORES=2
OMP_THREADS=$((CPUS_PER_TASK - RESERVED_CORES))

export OMP_NUM_THREADS=$OMP_THREADS
export MKL_NUM_THREADS=$OMP_THREADS

echo "Using $WORLD_SIZE GPUs | OMP Threads: $OMP_THREADS per process"

# Define learning rates, weight decays, batch sizes, and skip_factor values to scan
LEARNING_RATES=(0.001 0.0015)
WEIGHT_DECAYS=(0.05 0.1)
SKIP_FACTORS=(0.1 0.3 0.5 0.7 0.9)
BATCH_SIZES=(2048)


# Calculate indices for learning rate, weight decay, batch size, and skip_factor
IDX_LR=$(( SLURM_ARRAY_TASK_ID / (${#WEIGHT_DECAYS[@]} * ${#BATCH_SIZES[@]} * ${#SKIP_FACTORS[@]} ) ))
IDX_WD=$(( (SLURM_ARRAY_TASK_ID / (${#BATCH_SIZES[@]} * ${#SKIP_FACTORS[@]})) % ${#WEIGHT_DECAYS[@]} ))
IDX_BS=$(( (SLURM_ARRAY_TASK_ID / ${#SKIP_FACTORS[@]}) % ${#BATCH_SIZES[@]} ))
IDX_SKIP=$(( SLURM_ARRAY_TASK_ID % ${#SKIP_FACTORS[@]} ))

echo "Computed indices: IDX_LR=${IDX_LR} => LR=${LEARNING_RATES[$IDX_LR]}, IDX_WD=${IDX_WD}, IDX_BS=${IDX_BS}, IDX_SKIP=${IDX_SKIP}"

# Configuration variables
MODULE_NAME="dna_linear_nested_tf"
CONFIG_PATH="/home/${USER}/export_vision/dna/configs/$MODULE_NAME.yaml"

restart_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'Restarts=\d+' | cut -d '=' -f 2)
RESUME_FLAG="False"
if [ $restart_count -gt 0 ]; then
  RESUME_FLAG="True"
fi

# Initialize return code
ret=0
launch_job() {
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    main_ddp.py \
    --config_path="$CONFIG_PATH" \
    --resume="$RESUME_FLAG" \
    --model.max_path_len=11 \
    --model.module_top_k=2 \
    --model.module_temperature=None \
    --model.n_tf=22 \
    --model.embed_dim=256 \
    --model.num_heads=4 \
    --model.skip_factor=${SKIP_FACTORS[$IDX_SKIP]} \
    --model.bias_u=0.0001 \
    --model.use_id=False \
    --model.use_bias=False \
    --model.early_exit=False \
    --model.hard_exit=False \
    --model.start_node_depth=1 \
    --model.in_block_rescale=True \
    --training.num_epochs=300 \
    --training.warmup_epochs=10 \
    --training.batch_size=${BATCH_SIZES[$IDX_BS]} \
    --training.learning_rate=${LEARNING_RATES[$IDX_LR]} \
    --training.weight_decay=${WEIGHT_DECAYS[$IDX_WD]} \
    --training.clip_norm=0.0 \
    --logging.save_dir="/tmp" \
    --logging.entity="${USER}" \
    --logging.project="tmp" \
    || ret=$?
}

# Launch job
launch_job
# Resubmit the job up to 5 times if it failed
if [ $ret -ne 0 ] && [ $restart_count -lt 10 ]; then
  echo "Task $SLURM_ARRAY_TASK_ID failed. Restart count: $restart_count. Requeuing..."
  scontrol requeue $SLURM_JOB_ID
fi