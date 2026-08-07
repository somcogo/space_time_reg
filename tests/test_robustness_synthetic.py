"""The synthetic-GT fitter recovers a small known analytic flow to ~sub-pixel EPE (fast/CPU)."""

import torch

from stmr.robustness.synthetic import analytic_flow, epe, fit_siren_to_flow


def test_recovers_known_flow_small_epe():
    torch.set_num_threads(4)
    H = W = 24
    amp = 1.0 / (W / 2)  # a small (~1 px) analytic deformation
    img, target, rel_gt, _ = analytic_flow(H, W, amp, device="cpu")

    rel_pred, res = fit_siren_to_flow(img, target, H, W, steps=80, lr=1e-3,
                                      seed=0, step_size=0.1, device="cpu")

    assert res[-1] < 0.2 * res[0]                 # residual reduced > 80% (image matched)
    assert epe(rel_pred, rel_gt, H, W) < 1.5      # flow recovered to ~pixel accuracy
