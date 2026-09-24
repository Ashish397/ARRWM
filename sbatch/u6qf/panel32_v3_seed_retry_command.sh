#!/bin/bash
set -euo pipefail
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
P=$R/panel32_v3_stage
rm -f "$P/logs/seed_setup/COMPLETE"
rm -f "$P/logs/seed_setup/build.log" "$P/logs/seed_setup/validate.log"
PANEL32_STAGE="$P" bash "$R/ARRWM/sbatch/u6qf/panel32_v3_seed_setup.sh"
