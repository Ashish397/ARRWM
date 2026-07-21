#!/bin/bash
# After the fetcher's first pass completes: run a second verify pass (re-pulls
# the deleted corrupt MG2 files), check integrity, resubmit both MG2 smokes.
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
MG=$ARR/third_party/Matrix-Game/Matrix-Game-2
LOG=$ARR/.monitor/fetch_weights.log
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

until grep -q "ALL DONE" $LOG; do sleep 60; done
echo "[regate] first pass done; running verify pass"
python $ARR/.monitor/fetch_weights.py 2>&1 | grep -E "OK|FAIL|ALL DONE" | tail -20

python - <<'EOF'
import os, zipfile, sys, json
from huggingface_hub import HfApi
MG = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0"
info = HfApi().model_info("Skywork/Matrix-Game-2.0", files_metadata=True)
bad = []
for s in info.siblings:
    p = os.path.join(MG, s.rfilename)
    if os.path.exists(p) and s.size and os.path.getsize(p) != s.size:
        bad.append((s.rfilename, os.path.getsize(p), s.size))
clip = os.path.join(MG, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
if not zipfile.is_zipfile(clip):
    bad.append(("clip zip check", "corrupt", ""))
if bad:
    print("[regate] STILL BAD:", bad); sys.exit(1)
print("[regate] MG2 integrity OK")
EOF
if [ $? -ne 0 ]; then echo "[regate] integrity failed - NOT resubmitting"; exit 1; fi

cd $ARR
J1=$(sbatch --parsable sbatch/matrixgame_smoke.sbatch); echo "[regate] mg2-smoke resubmitted as $J1"
J2=$(sbatch --parsable sbatch/mg2base_smoke.sbatch); echo "[regate] mg2base-smoke resubmitted as $J2"
