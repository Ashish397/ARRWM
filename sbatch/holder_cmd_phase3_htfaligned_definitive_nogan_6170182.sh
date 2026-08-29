#!/bin/bash
set -euo pipefail
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=6170182 NODE=nid010208 PORTOFF=31982 \
RUNSTAMP=2808p3htf_nogan_h182_r3 \
bash sbatch/run_phase3_htfaligned_definitive_nogan_node.sh
