from argparse import Namespace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.tensorboard import SummaryWriter

from src.registration import registration
from src.models.factory import get_func
from src.data.data_load import prepare_inputs

def train(config):
    func = get_func(config.func_name, config.func_kwargs)
    func = func.to(config.device)
    writer = SummaryWriter(log_dir=os.path.join(config.log_path, 'tensorboard'))
    logger = None
    inputs, eval_inputs = prepare_inputs(config, logger)
    out = registration(config=config,
                       writer=writer,
                       logger=logger,
                       inputs=inputs,
                       eval_inputs=eval_inputs,
                       func=func)

def main(**kwargs):
    config = Namespace(**kwargs)
    config.func_kwargs = {
        'layers': [2, 64, 64, 64, 2],
        'weight_init': True,
        'last_init_zero': False,
        'omega': 30,
        # 'img_sz': (128, 128),
        # 'smoothing_kernel': 'AK',
        # 'smoothing_win': 15,
        # 'smoothing_pass': 1,
        # 'ds': 2,
        # 'bs': 16,
        # 'use_t': False,
    }
    config.log_path = os.path.join('log/motion_test', config.comment)
    train(config)

if __name__ == '__main__':
    comment = 'first_test'
    debug = False

    epochs = 500
    lr=1e-4
    solver = 'euler'
    step_size = 0.1
    func_name = 'siren'
    lambda_st = 1
    lambda_grd = 1e-3
    lambda_negJ = 1e-3
    lambda_hel = 0.
    lambda_pgr = 0.
    lambda_lap = 0.
    lambda_recon = 0.
    weight_decay = 0.
    start_frame = 0
    time_points = 20
    schedule = [1]

    dataset = 'rot_slow2'
    device='cuda'
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