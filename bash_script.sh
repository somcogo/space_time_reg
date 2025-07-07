#!/bin/bash
lossfn="mse"
depth="3"
dim="64"
epochs="500"
tp="20"
ts=0

dset="rot_slow2"
gpu="6"
solver="dopri5"
netw="siren"

folder="loss_test"
comm="detJ_loss_fin"

lr="1e-4"
lam_negJ=1e-2
lam_smt=0
lam_grd=0
lam_lap=0
lam_pgr=0
lam_sim=1

atol=1e-8
rtol=1e-6

seed="0"
fin_diff_grad=no-
use_grid=
autograd_grid=


python main.py --epochs=$epochs --exp_name="${folder}/${comm}-${dset}-lr${lr}-negJ${lam_negJ}-smt${lam_smt}-grd${lam_grd}-lap${lam_lap}-pgr${lam_pgr}-sim${lam_sim}-dep${depth}-dim${dim}-tp_${tp}-e${epochs}-${solver}-ts${ts}-${netw}-${fin_diff_grad}findif-${autograd_grid}aggrid-${use_grid}grid-rtol${rtol}-atol${atol}"  --log_cadence=50 --lr=$lr --dataset=$dset --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd --lambda_st=$lam_sim --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver --func_name=$netw --seed=$seed --rtol=$rtol --atol=$atol --${fin_diff_grad}fin_diff_grad --${use_grid}use_grid --${autograd_grid}autograd_grid --step_size=$ts --lambda_lap=$lam_lap --lambda_pgr=$lam_pgr