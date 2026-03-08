#!/bin/bash
#SBATCH --job-name=olmo3-7b
#SBATCH --partition=general
#SBATCH --nodes=8
#SBATCH --exclude=pi1-h100-25,pi1-h100-27
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/shaoliang/logs/olmo_%j/olmo3-7b-%j.out
#SBATCH --error=/home/shaoliang/logs/olmo_%j/olmo3-7b-%j.err
#SBATCH --exclusive

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
OLMO_DIR="/home/shaoliang/OLMo-core"
SAVE_DIR_BASE="/home/shaoliang/olmo_checkpoints/olmo3-7b"
SAVE_DIR="${SAVE_DIR_BASE}/job-${SLURM_JOB_ID:-manual}"
WORK_DIR="/home/shaoliang/dataset-cache"
CONDA_ENV="olmo"
SCRIPT="src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py"

mkdir -p "$SAVE_DIR" "$WORK_DIR" /home/shaoliang/logs

# ── Environment ────────────────────────────────────────────────────────────────
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Rendezvous: resolve master node to its private 10.0.1.x IP
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(ssh "$MASTER_HOSTNAME" "ip addr show ens10f0np0 | grep 'inet 10\.' | awk '{print \$2}' | cut -d/ -f1")
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT
export OLMO_DIST_BACKEND=nccl
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens10f0np0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=100000
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_ENABLE_MONITORING=1

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Node list:     $SLURM_JOB_NODELIST"
echo "GPUs/node:     $SLURM_GPUS_PER_NODE"
echo "Master host:   $MASTER_HOSTNAME"
echo "Master IP:     $MASTER_ADDR:$MASTER_PORT"
echo "Working dir:   $OLMO_DIR"
echo "Save base dir: $SAVE_DIR_BASE"
echo "Save dir:      $SAVE_DIR"
echo "Overrides:     --train_module.compile_model=true --train_module.rank_microbatch_size=8192 --data_loader.global_batch_size=524288 --checkpointer.save_interval=10000"
echo "============================================"

# ── GPU cleanup ─────────────────────────────────────────────────────────────────
# Kill any zombie GPU processes left over from previous crashed runs.
echo "Cleaning up stale GPU processes on all nodes..."
srun bash -c '
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
if [ -n "$PIDS" ]; then
    echo "Node $(hostname): killing stale GPU PIDs: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
else
    echo "Node $(hostname): GPUs clean"
fi
'

# ── Launch ─────────────────────────────────────────────────────────────────────
# Each node resolves its own private IP via the wrapper, then passes it to
# torchrun --local-addr so peers are advertised by IP instead of unresolvable FQDN.
srun bash -c '
LOCAL_IP=$(ip addr show ens10f0np0 | grep "inet 10\." | awk "{print \$2}" | cut -d/ -f1)
echo "Node $(hostname): local_ip=$LOCAL_IP"
exec torchrun \
    --nnodes='"$SLURM_JOB_NUM_NODES"' \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
    --rdzv-id='"$SLURM_JOB_ID"' \
    --local-addr="$LOCAL_IP" \
    '"$OLMO_DIR/$SCRIPT"' \
    --name='"olmo3-7b-${SLURM_JOB_ID}"' \
    --save-folder='"$SAVE_DIR"' \
    --work-dir='"$WORK_DIR"' \
    --train_module.compile_model=true \
    --train_module.rank_microbatch_size=8192 \
    --data_loader.global_batch_size=524288 \
    --checkpointer.save_interval=10000
'
