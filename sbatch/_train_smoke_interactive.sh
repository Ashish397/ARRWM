set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
python action-forcing/train.py \
  --config configs/action_ode_distill_F.yaml \
  chunked_lmdb=true auto_resume=false \
  clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_gt \
  cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_gt \
  clean_only=false require_cf=false lambda_cf=0.0 \
  random_steps="[0,15,18,19]" \
  generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
  max_steps=5 checkpoint_interval=1000 \
  logdir=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_train_smoke_fresh \
  config_name=ode14e_train_smoke
echo "TRAIN-SMOKE-INT done"
