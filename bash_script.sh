#!/bin/bash
lossfn="mse"
depth="3"
dim="64"
epochs="10"
recon_epochs="500"
tp="5"
ts=0.1
slice=0

dset="cmr_P001_Acc04"
solver="euler"
netw="sirenensemble"

folder="cmr/debug_init_recon"
comm="wrapper_scaling_search"

lr=1e-4 #1e-4 1e-6
init_lr=1e-2
recon_lr=1e-5 #1e-5 1e-5
weight_decay=0
lam_negJ=1e-2 #1e-2 1e-4
lam_grd=1e-4 #1e-4 1e-3
lam_lap=0
lam_pgr=0
lam_hel=0
lam_rec=1e-6 #1e-6 1e-6
lam_init_rec=1e-1
lam_sim=1

recon_scale=10

atol=1e-8
rtol=1e-6

seed="0"
fin_diff_grad=
use_grid=
autograd_grid=

debug=no-
nrep=no-
nmapg=
rand_mask=no-

start_frame=0

export CUDA_VISIBLE_DEVICES="4"

for recon_scale in -1 -2 -3 -4
do
python main.py --epochs=$epochs --recon_epochs=$recon_epochs --exp_name="${folder}/${comm}-${dset}-lr${lr}-rlr${recon_lr}-ilr${init_lr}-wd${weight_decay}-negJ${lam_negJ}-grd${lam_grd}-lap${lam_lap}-pgr${lam_pgr}-hel${lam_hel}-rec${lam_rec}-irec${lam_init_rec}-sim${lam_sim}-dep${depth}-dim${dim}-tp${tp}-e${epochs}-re${recon_epochs}-rs${recon_scale}-${solver}-ts${ts}-${netw}-${fin_diff_grad}findif-${autograd_grid}aggrid-${use_grid}grid-rtol${rtol}-atol${atol}-${nrep}nrep-st${start_frame}-sl${slice}-seed${seed}-${debug}db-${nmapg}apg-${rand_mask}rm"  --log_cadence=50 --lr=$lr --recon_lr=$recon_lr --init_lr=$init_lr --recon_scale=$recon_scale --weight_decay=$weight_decay --dataset=$dset --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_grd=$lam_grd --lambda_st=$lam_sim --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver --func_name=$netw --seed=$seed --rtol=$rtol --atol=$atol --${fin_diff_grad}fin_diff_grad --${use_grid}use_grid --${autograd_grid}autograd_grid --step_size=$ts --lambda_lap=$lam_lap --lambda_pgr=$lam_pgr --lambda_hel=$lam_hel --lambda_recon=$lam_rec --lambda_init_recon=$lam_init_rec --${debug}debug --${nrep}use_nreps --${nmapg}use_nmapg --start_frame=$start_frame --slice_number=$slice --${rand_mask}random_mask;
done