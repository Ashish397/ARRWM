#!/bin/bash
set -euo pipefail
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=6170183 NODE=nid010243 PORTOFF=31983 \
RUNSTAMP=2808p3htf_h183_r3 \
bash sbatch/run_phase3_htfaligned_definitive_node.sh
