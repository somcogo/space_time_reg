#!/bin/bash
dset="const"
lossfn="ngf"
gpu="6"
depth="2"
dim="64"
# lr="1e-5"
tp="20"
epochs="2000"
solver="euler"
ts="0.1"

comm="ngf_const"
for lr in "1e-6"
do
    for lam_negJ in 1e3
    do
        for lam_smt in 10
        do
            for lam_grd in 1
            do
                python main.py --epochs=$epochs --exp_name="after_ngf_fix/${comm}_${dset}_lr${lr}_negJ${lam_negJ}_smt${lam_smt}_grd${lam_grd}_dep${depth}_dim${dim}_tp_${tp}_e${epochs}_${solver}_ts${ts}"  --log_cadence=500 --lr=$lr --dataset=$dset --time_step=$ts --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver
            done
        done
    done
done