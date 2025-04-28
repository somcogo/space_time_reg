#!/bin/bash
dset="rot_slow2"
lossfn="ngf"
gpu="7"
depth="2"
dim="32"
# lr="1e-5"
tp="5"
epochs="10"

comm="x_zero_init"
for lr in "1e-5"
do
    for lam_negJ in 1e3
    do
        for lam_smt in 10
        do
            for lam_grd in 1
            do
                python main.py --epochs=$epochs --exp_name="${lossfn}_fix/${comm}_${dset}_lr${lr}_ts01_negJ${lam_negJ}_smt${lam_smt}_grd${lam_grd}_dep${depth}_dim${dim}_tp_${tp}_e${epochs}"  --log_cadence=500 --lr=$lr --dataset=$dset --time_step=0.1 --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp
            done
        done
    done
done