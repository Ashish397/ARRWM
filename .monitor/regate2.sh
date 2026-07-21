#!/bin/bash
# Pass-2 gate: after fetcher pass 1, delete ALL size-mismatched files across the
# plan (oversized frankenfiles can't self-heal: resume -> 416 -> false OK),
# re-run the fetcher, deep-verify, resubmit the three failed smokes.
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
LOG=$ARR/.monitor/fetch_weights.log
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

until grep -q "ALL DONE" $LOG; do sleep 60; done
echo "[regate2] pass 1 done; sweeping size mismatches"

python - <<'EOF'
import os
from huggingface_hub import HfApi
T = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party"
REPOS = {
    "Skywork/Matrix-Game-2.0": f"{T}/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0",
    "OpenDriveLab/Vista": f"{T}/Vista/ckpts",
    "tencent/HunyuanVideo-1.5": f"{T}/HY-WorldPlay/weights/HunyuanVideo-1.5",
    "google/byt5-small": f"{T}/HY-WorldPlay/weights/HunyuanVideo-1.5/text_encoder/byt5-small",
    "tencent/HY-WorldPlay": f"{T}/HY-WorldPlay/weights",
    "google/siglip-so400m-patch14-384": f"{T}/HY-WorldPlay/weights/_siglip_raw",
    "stdstu123/Yume-5B-720P": f"{T}/YUME/Yume-5B-720P",
    "stdstu123/Yume-I2V-540P": f"{T}/YUME/Yume-I2V-540P",
    "OpenGVLab/InternVL3-2B-Instruct": f"{T}/YUME/InternVL3-2B-Instruct",
    "OpenGVLab/InternVL3-8B": f"{T}/_vlm/InternVL3-8B",
    "ByteDance/Sa2VA-4B": f"{T}/_vlm/Sa2VA-4B",
}
api = HfApi()
for repo, root in REPOS.items():
    try:
        sibs = api.model_info(repo, files_metadata=True).siblings
    except Exception as e:
        print(f"[regate2] list fail {repo}: {str(e)[:80]}"); continue
    for s in sibs:
        p = os.path.join(root, s.rfilename)
        if os.path.exists(p) and s.size and os.path.getsize(p) != s.size:
            print(f"[regate2] DELETE mismatched {repo}/{s.rfilename} "
                  f"({os.path.getsize(p)} vs {s.size})")
            os.remove(p)
print("[regate2] sweep complete")
EOF

echo "[regate2] running fetch pass 2"
python $ARR/.monitor/fetch_weights.py 2>&1 | grep -E "FAIL|ALL DONE" | tail -5

python - <<'EOF'
import os, sys, zipfile
from safetensors import safe_open
T = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party"
bad = []
mg = f"{T}/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0"
for p in (f"{mg}/base_distilled_model/base_distill.safetensors",
          f"{mg}/base_model/diffusion_pytorch_model.safetensors",
          f"{T}/Vista/ckpts/vista.safetensors"):
    try:
        with safe_open(p, framework="pt") as f:
            next(iter(f.keys()))
    except Exception as e:
        bad.append((p, str(e)[:80]))
clip = f"{mg}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
if not zipfile.is_zipfile(clip):
    bad.append((clip, "not a zip"))
if bad:
    for b in bad: print("[regate2] STILL BAD:", b)
    sys.exit(1)
print("[regate2] deep verify OK (MG2 x3 + Vista)")
EOF
if [ $? -ne 0 ]; then echo "[regate2] verify failed - NOT resubmitting"; exit 1; fi

cd $ARR
echo "[regate2] resubmitting: mg2=$(sbatch --parsable sbatch/matrixgame_smoke.sbatch) mg2base=$(sbatch --parsable sbatch/mg2base_smoke.sbatch) vista=$(sbatch --parsable sbatch/vista_smoke.sbatch)"
