#!/bin/bash
dset="easy"
lossfn="mse"
gpu="3"
depth="3"
dim="64"
tp="5"
epochs="2000"
solver="euler"
ts="0.01"
netw="siren"

folder="autograd"

comm="siren_logreluJdet_oldreg"
lr="1e-4"
lam_negJ=1
lam_smt=0
# lam_grd=100
lam_sim=1
for lam_grd in 1 1e2 1e4 1e6 1e8
do
    python main.py --epochs=$epochs --exp_name="${folder}/${comm}-${dset}-lr${lr}-negJ${lam_negJ}-smt${lam_smt}-grd${lam_grd}-sim${lam_sim}-dep${depth}-dim${dim}-tp_${tp}-e${epochs}-${solver}-ts${ts}-${netw}"  --log_cadence=500 --lr=$lr --dataset=$dset --time_step=$ts --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd --lambda_st=$lam_sim --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver --func_name=$netw
done