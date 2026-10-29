"""Known-motion test data, motion metrics, and the related config/visualisation fixes
(CPU, no data files)."""

import numpy as np
import pytest
import torch
from flow_vis import flow_to_color

from stmr.ablation.runner import build_config
from stmr.ablation.stages import LADDERS
from stmr.config import Config
from stmr.data.known_motion import bright_region, synthesize
from stmr.metrics.motion_metrics import endpoint_error, inverse_consistency, magnitude
from stmr.metrics.prep_visuals import prep_flow_vis
from stmr.utils.spatial_transformer import GridSampleTransformer
from stmr.utils.spatial_utils import generate_coord_tensor

from _phantom import moving_phantom


def _cfg(**over):
    base = dict(device="cpu", dataset="cmr_test2", time_points=3, synth_flow="analytic",
                synth_flow_amp=1.0, synth_noise=0.0, synth_gain=0.0)
    base.update(over)
    return Config(**base).finalize()


def _frames(T=3, H=24, W=40):
    gt, _ = moving_phantom(T=T, H=H, W=W, shift_px=1.0)
    return gt + 1e-3 * torch.randn_like(gt)  # some background noise so the air estimate is finite


def test_synthesized_frames_follow_the_known_motion():
    torch.manual_seed(0)
    gt = _frames()
    frames, phi = synthesize(_cfg(), gt)
    assert phi.shape == (2, 24 * 40, 2)
    assert torch.equal(frames[0], gt[0])
    for k in range(2):  # frame_{k+1} = warp(frame_k, phi_k): phi is the exact frame-to-frame motion
        warped = GridSampleTransformer(phi[k:k + 1], [-1, *gt.shape[2:], 2]).apply(frames[k:k + 1])
        assert torch.allclose(frames[k + 1:k + 2], warped, atol=1e-6)


def test_synth_gain_brightens_only_the_bright_region():
    torch.manual_seed(0)
    gt = _frames()
    plain, _ = synthesize(_cfg(synth_flow_amp=0.0), gt)
    gained, _ = synthesize(_cfg(synth_flow_amp=0.0, synth_gain=0.1), gt)
    w = bright_region(gt[0])[0, 0]
    ratio = magnitude(gained)[2] / magnitude(plain)[2]
    assert torch.allclose(ratio[w == 1], torch.tensor(1.2), rtol=1e-5)  # 1 + synth_gain * t, t = 2
    assert torch.equal(gained[:, :, w == 0], plain[:, :, w == 0])  # untouched away from it
    assert w[~(magnitude(gt)[0] > 0.5 * magnitude(gt)[0].max())].max() < 1  # peak is on the blob


def test_synth_noise_is_reproducible_and_scaled():
    gt = _frames()
    a, _ = synthesize(_cfg(synth_noise=1.0), gt)
    b, _ = synthesize(_cfg(synth_noise=1.0), gt)
    c, _ = synthesize(_cfg(synth_noise=0.0), gt)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_endpoint_error_is_zero_for_the_truth_and_the_motion_for_zero_flow():
    gt = _frames()
    _, phi = synthesize(_cfg(synth_flow_amp=2.0), gt)
    coord = generate_coord_tensor((24, 40), "cpu")
    exact = endpoint_error(phi, phi, gt)
    assert exact["epe_crop"] == pytest.approx(0.0, abs=1e-6)
    zero = endpoint_error(coord.expand(2, -1, -1), phi, gt)
    assert zero["epe_crop"] == pytest.approx(zero["gt_motion_crop"], rel=1e-5)
    assert zero["epe_rel_crop"] == pytest.approx(1.0, rel=1e-5)


def test_inverse_consistency_of_exact_inverses_is_zero():
    gt = _frames(T=3, H=40, W=64)
    coord = generate_coord_tensor((40, 64), "cpu")
    shift = torch.tensor([1.0 / 19.5, -2.0 / 31.5])  # (1 px, -2 px)
    fwd = (coord + shift).expand(2, -1, -1)
    bwd = (coord - shift).expand(2, -1, -1)
    out = inverse_consistency(fwd, bwd, gt)
    assert out["fwd_bwd_crop"] < 1e-4
    unrelated = inverse_consistency(fwd, fwd, gt)  # same direction twice: error = 2x the motion
    assert unrelated["fwd_bwd_rel_crop"] == pytest.approx(2.0, rel=1e-3)


def test_stage_sourced_motion_needs_that_stage_in_the_outdir(tmp_path):
    s0r = {s.id: s for s in LADDERS["motion"]}["S0R"]
    with pytest.raises(FileNotFoundError, match="run S0"):
        build_config(s0r, 1, "cpu", str(tmp_path))
    (tmp_path / "S0").mkdir()
    torch.save({"phi": torch.zeros(1)}, tmp_path / "S0" / "res.pt")
    cfg = build_config(s0r, 1, "cpu", str(tmp_path), {"seed": 7})
    assert cfg.synth_flow == str(tmp_path / "S0" / "res.pt")
    assert cfg.seed == 8  # never the same initial flow as the stage whose motion it re-applies


def test_saved_config_with_removed_keys_still_loads():
    saved = {**Config().to_dict(), "hard_dc": True}
    with pytest.raises(ValueError):
        Config.from_dict(saved)
    with pytest.warns(UserWarning, match="hard_dc"):
        cfg = Config.from_saved(saved)
    assert cfg.dataset == Config().dataset


def test_flow_colours_use_horizontal_channel_as_u():
    """A purely horizontal displacement (channel 1 of phi) must get flow_vis' colour for u."""
    rel_phi = torch.zeros(1, 8, 12, 2)
    rel_phi[..., 1] = 0.1
    uv = np.zeros((8, 12, 2), dtype=np.float32)
    uv[..., 0] = 1.0
    img = prep_flow_vis(rel_phi)
    assert torch.equal(img[2, 2], torch.from_numpy(flow_to_color(uv)[0, 0]))


def test_fixed_motion_outputs():
    from stmr.registration import get_fixed_motion_outputs
    from stmr.state import EvalInputs, Inputs

    gt = _frames()
    _, phi = synthesize(_cfg(), gt)
    inputs = Inputs(moving=gt.clone(), moving_inr=None, fixed=None, forward=None,
                    phi_shape=[-1, 24, 40, 2])
    known = EvalInputs(init_recon=gt, gt_im=gt, seg_moving=None, seg_fixed=None, gt_phi=phi)
    coord = generate_coord_tensor((24, 40), "cpu")

    ident = get_fixed_motion_outputs(_cfg(fixed_motion="identity"), inputs, known)
    assert torch.equal(ident.abs_phi, coord.expand(2, -1, -1))
    assert torch.allclose(ident.transformer.apply(gt[:-1]), gt[:-1], atol=1e-6)
    oracle = get_fixed_motion_outputs(_cfg(fixed_motion="gt"), inputs, known)
    assert torch.equal(oracle.abs_phi, phi)
    assert torch.allclose(oracle.rel_vel, phi - coord)
    with pytest.raises(ValueError, match="synth_flow"):
        get_fixed_motion_outputs(_cfg(fixed_motion="gt"), inputs,
                                 EvalInputs(init_recon=gt, gt_im=gt, seg_moving=None, seg_fixed=None))
