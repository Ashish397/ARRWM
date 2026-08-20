source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=$PWD:$PWD/action-forcing
export TMPDIR=/tmp
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export ARRWM_ACTION_ENCODER=pca_raw
python utils/.probe_rope_boundary.py --buffer_frames 36 --window_frames 24 --chunks 16
