#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Exact htfhu37j strong/stable reference GAN, changing only the action-critic
# treatment.  Node 0 doubles the frozen guidance strength from .3 to .6.
# Node 1 makes the critic genuinely online: current teacher targets, DDP
# trainable critic, two critic updates, and the pca_raw target space used by
# the commanded actions and the pretrained v14e critic.
HOLDER=6170182 NODE=nid010187 PIXW=.0175 \
TARGET_SHARE=htfhu37j_rerun_acfrozen_strongg06 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
GAN_ACTION_COND=false GAN_ACTION_MISMATCH_WEIGHT=0 \
ACTION_CRITIC_FREEZE=true ACTION_CRITIC_GUIDANCE_WEIGHT=.6 \
ACTION_TEACHER_MODE=off ACTION_CRITIC_UPDATES=1 \
ACTION_CRITIC_Z_LOSS_WEIGHT=.25 ACTION_CRITIC_LR=3e-4 \
TEACHER_ACTION_ENCODER=pca_raw \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=y9ry8jhr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99_htfhu37j_acfrozen_strongg06 PORTOFF=29911 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_htfhu37j_acfrozen_strongg06_h6170182_nid010187.log 2>&1 &
pid0=$!

HOLDER=6170182 NODE=nid010208 PIXW=.0175 \
TARGET_SHARE=htfhu37j_rerun_aconline CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
GAN_ACTION_COND=false GAN_ACTION_MISMATCH_WEIGHT=0 \
ACTION_CRITIC_FREEZE=false ACTION_CRITIC_GUIDANCE_WEIGHT=.3 \
ACTION_TEACHER_MODE=all ACTION_CRITIC_UPDATES=2 \
ACTION_CRITIC_Z_LOSS_WEIGHT=.5 ACTION_CRITIC_LR=3e-4 \
TEACHER_ACTION_ENCODER=pca_raw \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=y9ry8jhr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99_htfhu37j_aconline PORTOFF=29912 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_htfhu37j_aconline_h6170182_nid010208.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
