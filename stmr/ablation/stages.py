"""The staged ablation ladder.

Each stage flips exactly one component on relative to the previous one, starting from
"fully-sampled GT images, registration only, no k-space undersampling" (S0) and building
up to the full ``configs/cmr_soft_con.yaml`` pipeline (S5). Running them in order and
reading off PSNR / the imdiff residual pinpoints which component first pulls the solution
off the ground truth.

Stages are expressed as ``BASE`` (constants held fixed across the ladder) + a per-stage
``overrides`` dict, materialised into a ``Config`` by :func:`stmr.ablation.runner.build_config`.
S5 loads the real YAML so the "full pipeline" stage is byte-for-byte the real experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# Knobs held constant across the ladder so that only the intended component changes per
# stage. Data/geometry (T, patient slice), the velocity net, the ODE solver and the base
# optimiser lrs all match configs/cmr_soft_con.yaml; imdiff_warp_mag is left at the Config
# default (matching the yaml) so S0-S4 are directly comparable to S5.
BASE = dict(
    device="cuda",
    seed=0,
    time_points=6,
    start_frame=0,
    slice_number=0,
    factor=4,
    mask="st",
    func_name="groupsiren",
    siren_depth=3,
    siren_dim=64,
    last_init_zero=True,
    solver="euler",
    step_size=0.1,
    atol=1e-8,
    rtol=1e-6,
    lr=1e-4,
    recon_lr=1e-5,
    recon_eps=1e-4,
    weight_decay=0.0,
    interval=0,
    motion_warmup=0,
    loss="mse",
    lambda_negJ=1e-2,
    lambda_grd=0.0,
    lambda_lap=0.0,
    lambda_pgr=0.0,
    lambda_hel=0.0,
    lambda_mcdc=0.0,
    log_cadence=50,
    debug=False,
)


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    overrides: dict
    primary_metric: str
    criterion: str
    expectation: str
    # (this_row, prev_row) -> bool. None means "report only" (no pass/fail gate). prev_row
    # is None for the first stage in a ladder.
    pass_fn: Optional[Callable[[dict, Optional[dict]], bool]] = None
    # If set, load Config.from_yaml(yaml) as the base instead of BASE, then apply overrides.
    yaml: Optional[str] = None


# --- pass/fail predicates -------------------------------------------------------------
# Thresholds on the anchoring stages are provisional: treat the first real (500-epoch) run
# as calibration and tighten them to the observed anchored values.

def _s0_pass(row, _prev):
    return row["imdiff_warp"] <= 0.7 * row["imdiff_base"] and (row["vel_rms"] or 0) > 0


def _anchor_pass(row, _prev):
    # Fully-sampled data consistency (image- or Fourier-space) must anchor the recon at GT.
    return row["full_psnr"] >= 40.0


def _fft_equiv_pass(row, prev):
    # The FFT works as intended: switching the sim loss from image space (prev, S1) to
    # Fourier space (this, S2) must reproduce the same anchored PSNR (Parseval). Also still
    # anchors at GT.
    return (prev is not None and row["full_psnr"] >= 40.0
            and abs(row["full_psnr"] - prev["full_psnr"]) <= 2.0)


def _prior_pass(row, prev):
    return prev is not None and (prev["full_psnr"] - row["full_psnr"]) <= 2.0


def _motion_pass(row, prev):
    return prev is not None and row["full_psnr"] >= prev["full_psnr"]


# Shared setup for the two fully-sampled "free recon" stages (S1 image-space sim, S2
# Fourier-space sim): identical except for sim_domain, so the switch between them isolates
# the FFT data-consistency path.
_FREE_RECON_FULLY_SAMPLED = dict(
    dataset="cmr_test2", init="gt", init_skip=True, learn_recon=True,
    hard_dc=False, lambda_st=1.0, lambda_recon=0.0, lambda_rl2=1e4,
)


STAGES = [
    StageSpec(
        id="S0",
        title="registration only (k-space fidelity off, recon frozen at GT)",
        overrides=dict(
            dataset="cmr_test2", init="gt", init_skip=True, learn_recon=False,
            hard_dc=False, lambda_st=0.0, lambda_recon=0.0, lambda_rl2=1e4,
        ),
        primary_metric="imdiff_warp",
        criterion="PASS if imdiff_warp <= 0.7 * imdiff_base and vel_rms > 0",
        expectation="warp(I_t) should match I_{t+1} clearly better than identity",
        pass_fn=_s0_pass,
    ),
    StageSpec(
        id="S1",
        title="+ free recon, fully-sampled, IMAGE-space sim loss",
        overrides={**_FREE_RECON_FULLY_SAMPLED, "sim_domain": "image"},
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB (calibrate on first run)",
        expectation="image-space data consistency anchors the recon at GT; reference for "
                    "the FFT check in S2",
        pass_fn=_anchor_pass,
    ),
    StageSpec(
        id="S2",
        title="+ free recon, fully-sampled, FOURIER-space sim loss (FFT check)",
        overrides={**_FREE_RECON_FULLY_SAMPLED, "sim_domain": "fourier"},
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB and |S2 - S1| <= 2 dB",
        expectation="only switch vs S1 is moving the sim loss to Fourier space; if the FFT "
                    "path is correct the PSNR matches S1 (Parseval)",
        pass_fn=_fft_equiv_pass,
    ),
    StageSpec(
        id="S3",
        title="+ WCRR image prior",
        overrides=dict(
            dataset="cmr_test2", init="gt", init_skip=True, learn_recon=True,
            hard_dc=False, lambda_st=1.0, lambda_rl2=1e4, sim_domain="fourier",
            lambda_recon=1.0, reg="learned", reg_variant="wcrr",
            reg_alpha=1.0, recon_scale=6.0,
        ),
        primary_metric="full_psnr",
        criterion="PASS if S2.full_psnr - S3.full_psnr <= 2 dB",
        expectation="quantifies how much the learned prior biases the solution at GT",
        pass_fn=_prior_pass,
    ),
    StageSpec(
        id="S4",
        title="+ undersampling + hard DC (start at GT)",
        overrides=dict(
            dataset="cmr_P001", init="gt", init_skip=True, learn_recon=True,
            hard_dc=True, lambda_st=1.0, lambda_rl2=1e4, sim_domain="fourier",
            lambda_recon=1.0, reg="learned", reg_variant="wcrr",
            reg_alpha=1.0, recon_scale=6.0,
        ),
        primary_metric="full_psnr",
        criterion="report-only: expect d_vs_init < 0 (drift off GT)",
        expectation="SUSPECTED BREAK -- under subsampled DC the objective minimum is away "
                    "from GT, so starting at GT should lose PSNR",
        pass_fn=None,
    ),
    StageSpec(
        id="S5",
        title="realistic nmAPG+WCRR init, motion OFF",
        overrides=dict(
            dataset="cmr_P001", init="adj", init_skip=False, use_nmapg=True,
            recon_epochs=150, init_lr=1e-2, init_loss="l2", lambda_init_recon=2e-2,
            learn_recon=True, hard_dc=True, lambda_st=1.0, lambda_rl2=0.0,
            lambda_recon=1.0, reg="learned", reg_variant="wcrr",
            reg_alpha=1.0, recon_scale=6.0,
        ),
        primary_metric="full_psnr",
        criterion="baseline (report)",
        expectation="CRR-only reference; realistic starting point without motion",
        pass_fn=None,
    ),
    StageSpec(
        id="S6",
        title="full pipeline (motion ON)",
        overrides={},
        yaml="configs/cmr_soft_con.yaml",
        primary_metric="full_psnr",
        criterion="PASS if S6.full_psnr >= S5.full_psnr",
        expectation="motion contribution = S6 - S5 (prior sweeps: ~+0.15 dB)",
        pass_fn=_motion_pass,
    ),
]

STAGE_BY_ID = {s.id: s for s in STAGES}


def resolve_stage_ids(spec_str: str) -> list[str]:
    """Parse a --stages argument: a range 'S0-S5', a comma list 'S0,S3,S5', or 'all'."""
    spec_str = spec_str.strip()
    if not spec_str or spec_str.lower() == "all":
        return [s.id for s in STAGES]
    if "-" in spec_str and "," not in spec_str:
        lo, hi = spec_str.split("-")
        order = [s.id for s in STAGES]
        i, j = order.index(lo.strip()), order.index(hi.strip())
        return order[i:j + 1]
    ids = [tok.strip() for tok in spec_str.split(",") if tok.strip()]
    for sid in ids:
        if sid not in STAGE_BY_ID:
            raise ValueError(f"Unknown stage {sid!r}; known: {list(STAGE_BY_ID)}")
    return ids
