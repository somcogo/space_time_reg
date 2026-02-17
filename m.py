from src.debug.cmr_gt_motion_ft_abs import main

if __name__ == '__main__':
    debug = True

    epochs = 100
    lr=1e-6
    solver = 'euler'
    step_size = 0.01
    func_name = 'sirenensemble'
    lambda_st = 1
    lambda_grd = 0
    lambda_negJ = 0.
    lambda_hel = 0.
    lambda_pgr = 0.
    lambda_lap = 0.
    lambda_recon = 0.
    weight_decay = 0.
    start_frame = 0
    time_points = 12
    schedule = [1]

    dataset = 'heart_gt_ft_abs'
    device='cuda'
    for lambda_grd in [0.]:
        # for lambda_negJ in [1e-2, 1e-3, 1e-4, 1e-5]:
            comment = f'first_try2-lr{lr}-grd{lambda_grd}-e{epochs}'
            main(
                log_cadence=50,
                epochs=epochs,
                lr=lr,
                schedule=schedule,
                solver=solver,
                use_nreps=False,
                loss='mse',
                debug=debug,
                atol=1e-8,
                rtol=1e-6,
                step_size=step_size,
                func_name=func_name,
                comment=comment,
                dataset=dataset,
                device=device,
                start_frame=start_frame,
                time_points=time_points,

                lambda_st=lambda_st,
                lambda_grd=lambda_grd,
                lambda_negJ=lambda_negJ,
                lambda_hel=lambda_hel,
                lambda_pgr=lambda_pgr,
                lambda_lap=lambda_lap,
                lambda_recon=lambda_recon,
                weight_decay=weight_decay,
            )