#!/bin/bash
# VERIFY CLAIM 2 (static / challenge setting): does motion help reconstruction when the
# undersampling mask is STATIC across time (as in CMRxRecon)?
#
# Init is the per-frame nmAPG+CRR reconstruction from static-masked k-space (the challenge-
# appropriate single-frame baseline). At matched settings it runs:
#   - crronly : main-loop refinement with CRR + hard data consistency, NO motion (lambda_rl2=0)
#   - motion  : same + temporal/motion coupling                                  (lambda_rl2=1e4)
# Motion contribution = PSNR(motion) - PSNR(crronly).
#
# Usage:  bash motion_findings/run_decomposition.sh <factor> [patient=cmr_P001] [epochs=2000]
set -e
factor=${1:-4}
dset=${2:-cmr_P001}
epochs=${3:-2000}
cd "$(dirname "$0")/.."

common="--recon_epochs=150 --motion_warmup=0 --log_cadence=500 --lr=1e-4 --recon_lr=1e-5 \
  --recon_eps=1e-4 --init_lr=1e-2 --recon_scale=6 --reg_variant=wcrr --reg_alpha=1 --weight_decay=0 \
  --dataset=$dset --loss=mse --lambda_negJ=1e-2 --lambda_grd=0 --lambda_st=1 --lambda_recon=1 \
  --siren_depth=3 --siren_dim=64 --time_points=6 --solver=euler --func_name=groupsiren --seed=0 \
  --rtol=1e-6 --atol=1e-8 --step_size=0.1 --lambda_lap=0 --lambda_pgr=0 --lambda_hel=0 \
  --lambda_init_recon=2e-2 --no-debug --no-use_nreps --use_nmapg --start_frame=0 --slice_number=0 \
  --learn_recon --interval=0 --last_init_zero --mask=st --hard_dc --factor=$factor --epochs=$epochs"

run () { # <gpu> <exp> <extra...>
  CUDA_VISIBLE_DEVICES=$1 python main.py $common --exp_name="cmr/soft_con/decomp_static/$2" ${@:3} \
    2>&1 | tail -2 &
}
run 0 f${factor}-crronly --lambda_rl2=0
run 1 f${factor}-motion  --lambda_rl2=1e4
wait
echo "STATIC DECOMPOSITION (factor $factor) DONE -- summarize with:"
echo "  python motion_findings/summarize_decomposition.py"
