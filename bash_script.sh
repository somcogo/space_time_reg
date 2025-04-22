#!/bin/bash
dset="rec"
lossfn="ngf"
for lam_negJ in 1e3 1e5 1e7
do
    for lam_smt in 1 10 100
    do
        for lam_grd in 0.1 1 10
        do
            python main.py --epochs=2000 --exp_name="${lossfn}/${dset}_lr-5_ts01_negJ${lam_negJ}_smt${lam_smt}_grd${lam_grd}_negJnew"  --log_cadence=500 --lr=1e-5 --dataset=$dset --time_step=0.1 --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_smt=$lam_smt --lambda_grd=$lam_grd
        done
    done
done