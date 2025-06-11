#!/bin/bash
dset="easy"
lossfn="mse"
gpu="3"
# depth="2"
# dim="32"
tp="5"
epochs="2000"
solver="euler"
ts="0.01"
netw="sirent"

folder="sirent"

comm="esqlogJdet"
lr="1e-4"
lam_negJ=1
lam_smt=10
lam_grd=1000
for dim in 64
do
    for depth in 3
    do
        python main.py --epochs=$epochs --exp_name="${folder}/${comm}-${dset}-lr${lr}-negJ${lam_negJ}-smt${lam_smt}-grd${lam_grd}-dep${depth}-dim${dim}-tp_${tp}-e${epochs}-${solver}-ts${ts}-${netw}"  --log_cadence=500 --lr=$lr --dataset=$dset --time_step=$ts --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver --func_name=$netw
    done
done