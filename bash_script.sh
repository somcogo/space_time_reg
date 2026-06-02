#!/bin/bash
lossfn="mse"
depth="3"
dim="64"
epochs="1000"
recon_epochs="50"
tp="2"
ts=0.1
slice=0

dset="cmr_P001"
solver="euler"
netw="groupsiren"

# folder="cmr/soft_con/new_W_search"
comm="short"

# lr=1e-4
init_lr=1e-2
recon_lr=1e-6
weight_decay=0
# lam_negJ=1e-10
lam_grd=1e-10
# lam_rl2=1e2
lam_lap=0
lam_pgr=0
lam_hel=0
# lam_rec=1e-5
lam_init_rec=1e-2
lam_sim=1

recon_scale=2
recon_alpha=1

atol=1e-8
rtol=1e-6

seed="0"

debug=no-
nrep=no-
nmapg=
rand_mask=no-
learn_recon=
vel_init_zero=

start_frame=0
factor=4
interval=0

export CUDA_VISIBLE_DEVICES="1"

# folder="cmr/soft_con/long_test"
for lam_negJ in 1e-1 1e-2 1e-3 1e-4 1e-5
do
for lr in 1e-3 1e-4 1e-5
do
for lam_rl2 in 1e2 1e1 1e0 1e-1 1e-2
do
for lam_rec in 1e-4 1e-5 1e-6
do
folder="cmr/soft_con/new_W_search_lr${lr}"
python main.py --epochs=$epochs --recon_epochs=$recon_epochs --exp_name="${folder}/${comm}-${dset}-lr${lr}-rlr${recon_lr}-ilr${init_lr}-wd${weight_decay}-negJ${lam_negJ}-grd${lam_grd}-rl2${lam_rl2}-lap${lam_lap}-pgr${lam_pgr}-hel${lam_hel}-rec${lam_rec}-irec${lam_init_rec}-sim${lam_sim}-dep${depth}-dim${dim}-tp${tp}-e${epochs}-re${recon_epochs}-rs${recon_scale}-${solver}-ts${ts}-${netw}-st${start_frame}-sl${slice}-seed${seed}-${debug}db-${nmapg}apg-${rand_mask}rm-f${factor}-a${recon_alpha}-${learn_recon}rec-int${interval}-${vel_init_zero}lit"  --log_cadence=50 --lr=$lr --recon_lr=$recon_lr --init_lr=$init_lr --recon_scale=$recon_scale --reg_alpha=$recon_alpha --weight_decay=$weight_decay --dataset=$dset --loss=$lossfn --lambda_negJ=$lam_negJ --lambda_grd=$lam_grd --lambda_rl2=$lam_rl2 --lambda_st=$lam_sim --gpu_number=$gpu --siren_depth=$depth --siren_dim=$dim --time_points=$tp --solver=$solver --func_name=$netw --seed=$seed --rtol=$rtol --atol=$atol --step_size=$ts --lambda_lap=$lam_lap --lambda_pgr=$lam_pgr --lambda_hel=$lam_hel --lambda_recon=$lam_rec --lambda_init_recon=$lam_init_rec --${debug}debug --${nrep}use_nreps --${nmapg}use_nmapg --start_frame=$start_frame --slice_number=$slice --${rand_mask}random_mask --factor=${factor} --${learn_recon}learn_recon --interval=${interval} --${vel_init_zero}last_init_zero;
done
done
done
done