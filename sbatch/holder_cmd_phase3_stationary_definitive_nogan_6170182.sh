#!/bin/bash
set -euo pipefail
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=6170182 NODE=nid010187 \
RUNSTAMP=2808phase3stationarydefinitive_nogan_h6170182 \
bash sbatch/run_phase3_stationary_definitive_nogan_node.sh
