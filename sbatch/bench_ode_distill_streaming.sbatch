#!/bin/bash
#SBATCH --job-name=bench_ode_distill_streaming
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
#
# Streaming-inference benchmark for action_ode_distill_10h step 300.
#
# Fixed:
#   chunks_per_step = 2  (6 frames per streaming step; denoising default)
#   denoising_steps = 4
#
# KV cache sweep:
#   Probe cache_chunks in {4, 8, 16} in order. At each we run 1 timed
#   streaming step (warmup first to fill the cache and hit steady state).
#   First size that OOMs triggers one fallback probe at int(0.75 * N)
#   and then we stop. cache_chunks=3 is not tested here; that datapoint
#   already exists from job 4139941 (chunks2_cache3).
#
# Each test launches its own Python process (fresh CUDA allocator) to
# avoid fragmentation carrying over between sizes.
#
# Output JSONs land in logs/bench_ode_distill_streaming_step300_<jobid>/

set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

_orig_args=("$@")
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
set -- "${_orig_args[@]}"

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------------------
STUDENT_CKPT="${1:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_10h/latest_0000300.pt}"
TEACHER_CKPT="${2:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt}"
CONFIG="${3:-configs/action_ode_distill.yaml}"

CHUNKS_PER_STEP=2
DENOISING_STEPS=4

OUT_DIR="/scratch/u6ex/as1748.u6ex/ARRWM/logs/bench_ode_distill_streaming_step300_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"
echo "=== $(date) job=${SLURM_JOB_ID:-local} ==="
echo "Student ckpt:     $STUDENT_CKPT"
echo "Teacher ckpt:     $TEACHER_CKPT"
echo "Config:           $CONFIG"
echo "chunks_per_step:  $CHUNKS_PER_STEP"
echo "denoising_steps:  $DENOISING_STEPS"
echo "Output dir:       $OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

COMMON_ARGS=(
    --ckpt "$TEACHER_CKPT"
    --student-ckpt "$STUDENT_CKPT"
    --config "$CONFIG"
    --denoising_steps "$DENOISING_STEPS"
    --chunks_per_step "$CHUNKS_PER_STEP"
)

# Timed portion = exactly 1 streaming step (= 2 base chunks).
# Warmup scaled so the KV cache is fully populated before the timed step:
# warmup_chunks = cache_chunks + chunks_per_step. This guarantees the
# sliding-window eviction branch has run at least once, so the timed step
# exercises steady-state attention over the full local_attn_size.
# bench CLI rounds warmup_chunks up to whole streaming steps internally.
TIMED_CHUNKS=$CHUNKS_PER_STEP      # 2  => 1 timed streaming step

# Runs one bench size. Returns the python process's exit code so the
# caller can branch on success / OOM.
run_one() {
    local cache_chunks="$1"
    local label="$2"
    local warmup_chunks=$(( cache_chunks + CHUNKS_PER_STEP ))
    local out="$OUT_DIR/${label}.json"
    echo
    echo "==================== $label  (cache_chunks=$cache_chunks, warmup=$warmup_chunks) ===================="
    python action-forcing/bin/bench_streaming_inference.py \
        "${COMMON_ARGS[@]}" \
        --label "$label" \
        --output "$out" \
        --cache_chunks "$cache_chunks" \
        --total_chunks "$TIMED_CHUNKS" \
        --warmup_chunks "$warmup_chunks"
    return $?
}

# -------------------------------------------------------------------
# Sweep cache_chunks ∈ {4, 8, 16}. Stop at first OOM and try 0.75 * N.
# -------------------------------------------------------------------
PROBE_ORDER=(4 8 16)
FAILED_AT=""

for CC in "${PROBE_ORDER[@]}"; do
    if run_one "$CC" "cache${CC}"; then
        echo "[sweep] cache_chunks=$CC  OK"
    else
        rc=$?
        echo "[sweep] cache_chunks=$CC  FAILED (exit=$rc) — attempting fallback."
        FAILED_AT=$CC
        break
    fi
done

if [ -n "$FAILED_AT" ]; then
    # Integer 0.75 * FAILED_AT, floor. E.g. 16 -> 12, 8 -> 6, 4 -> 3.
    FALLBACK=$(( FAILED_AT * 3 / 4 ))
    if [ "$FALLBACK" -lt 1 ]; then FALLBACK=1; fi
    echo
    echo "[sweep] Fallback: trying cache_chunks=$FALLBACK (0.75 * $FAILED_AT)."
    if run_one "$FALLBACK" "cache${FALLBACK}_fallback_from_${FAILED_AT}"; then
        echo "[sweep] fallback cache_chunks=$FALLBACK  OK — stopping here per spec."
    else
        rc=$?
        echo "[sweep] fallback cache_chunks=$FALLBACK ALSO failed (exit=$rc). Stopping."
    fi
fi

# -------------------------------------------------------------------
echo
echo "==================== Summary ===================="
python - <<PY
import json, glob, os
from pathlib import Path
out_dir = Path("$OUT_DIR")
rows = []
for p in sorted(out_dir.glob("*.json")):
    try:
        entries = json.loads(p.read_text())
        for e in entries:
            rows.append({
                "file":            p.name,
                "label":           e.get("label", ""),
                "cache_chunks":    e.get("cache_chunks"),
                "chunks_per_step": e.get("chunks_per_step"),
                "local_attn":      e.get("local_attn_size_frames"),
                "chunks_out":      e.get("total_chunks_emitted"),
                "per_chunk_ms":    round(e.get("per_chunk_ms_mean", 0.0), 1),
                "per_chunk_p95":   round(e.get("per_chunk_ms_p95", 0.0), 1),
                "per_frame_ms":    round(e.get("per_frame_ms_mean", 0.0), 1),
                "per_fwd_ms":      round(e.get("per_forward_ms_mean", 0.0), 1),
                "peak_gib":        round(e.get("peak_mem_gib", 0.0), 2),
            })
    except Exception as exc:
        print(f"[WARN] could not parse {p}: {exc}")

if not rows:
    print("(no results)")
else:
    keys = ["label","cache_chunks","chunks_per_step","local_attn",
            "chunks_out","per_chunk_ms","per_chunk_p95","per_frame_ms",
            "per_fwd_ms","peak_gib"]
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))
PY

echo
echo "=== $(date) bench_ode_distill_streaming finished ==="
