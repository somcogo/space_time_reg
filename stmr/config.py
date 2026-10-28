"""Typed experiment configuration.

Replaces the previous 61-flag argparse block and the in-place ``config.*`` mutation
scattered through ``main.py``. A run is fully described by a ``Config``; experiments live
as YAML files under ``configs/`` and override only the fields they care about.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class Config:
    # --- experiment / logging ---
    exp_name: str = "test/test"
    log_path: str = "log"
    seed: int = 0
    device: str = "cuda"
    log_level: str = "info"
    log_cadence: int = 10
    metric_cadence: int = 1  # eval metrics, TensorBoard scalars and the summary-pdf curves every N
    #                          epochs (plus the first and last); they cost ~0.15-0.5 s/epoch on CPU
    debug: bool = False

    # --- data ---
    dataset: str = "cmr_P001"
    time_points: int = 20
    start_frame: int = 0
    slice_number: int = 0
    factor: int = 4
    mask: str = "st"
    random_mask: bool = False
    template_frames: int = 0  # leading frames kept fully-sampled + recon frozen at GT (0 = off)
    normalize_data: bool = True  # divide k-space by the max GT magnitude over all loaded frames
    cmr_raw: bool = False  # cmr_P*: load the challenge's own single-coil .mat (data/raw/cmrxrecon)
    #                        instead of the processed _norm file = it x a real scale per patient x
    #                        a per-frame sign; normalize_data removes the scale, the sign is invisible
    reverse_frames: bool = False  # load the frames in reverse time order (forward/backward motion check)

    # --- known-motion test (stmr/data/known_motion.py; off while synth_flow is '') ---
    # The GT frames are replaced by frame 0 warped with a known motion, so the learned motion can
    # be scored against it (ablation stages S0K / S0R).
    synth_flow: str = ""  # '' off | 'analytic' | path to a res.pt whose 'phi' is the known motion
    #                       | 'stage:<id>' (that stage's res.pt in the same ablation outdir)
    synth_flow_amp: float = 1.0  # analytic: displacement amplitude in px; res.pt: displacement multiplier
    synth_noise: float = 1.0  # fresh noise per frame, in units of the noise std measured in the data
    synth_gain: float = 0.0  # known non-motion brightening of the bright (blood-pool-like) region per frame

    # --- initial reconstruction ---
    init: str = "adj"
    init_skip: bool = False
    use_nmapg: bool = True
    recon_epochs: int = 300
    init_lr: float = 0.1
    init_loss: str = "l2"
    init_reg_abs: bool = False
    init_grad: str = "autograd"  # 'autograd' or 'analytic' (closed form; needs init_loss='l2' and reg='learned')
    tol: float = 0.0  # nmAPG stop on relative change; 0 = always run recon_epochs iterations
    L_init: float = 1.0  # nmAPG's initial Lipschitz estimate (first step = 1/L_init); independent of lambda_st
    detach_grads: bool = True
    lambda_init_recon: float = 0.2
    kspace_noise_sigma: float = 0.0  # robustness: relative noise on measured k-space (0=off)
    init_noise_sigma: float = 0.0    # robustness: relative noise on the init recon (0=off)

    # --- reconstruction regularizer ---
    reg: str = "learned"
    reg_variant: str = "crr"  # 'crr' (weak_convexity=0) or 'wcrr' (weak_convexity=1)
    reg_alpha: float | None = 1.0  # None -> keep the pretrained alpha (do not overwrite)
    reg_beta: float | None = 3.0  # None -> keep the pretrained beta (CRR 4.28, WCRR 1.15); lower = wider quadratic zone, less stiff
    recon_scale: float = 3.55  # 0 -> None (keep pretrained scale)

    # --- velocity network ---
    func_name: str = "groupsiren"  # F5: default to the non-stationary ensemble
    siren_depth: int = 3
    siren_dim: int = 256
    siren_omega: int = 30
    last_init_zero: bool = True
    last_init_std: float = 1e-6  # with last_init_zero: >0 -> last layer ~ N(0, std^2) (1e-6 ~ 1e-3 px
    #                              initial motion); 0 -> U(+-1/600), which already moves ~1 px

    # --- ODE solver ---
    solver: str = "rk4"
    step_size: float = 0.001  # 0 -> adaptive (None)
    atol: float = 1e-9
    rtol: float = 1e-7
    adjoint: bool = True  # odeint_adjoint (O(1) memory in steps) vs plain odeint: direct backprop,
    #                       the exact gradient of the discrete solve, memory grows with the steps

    # --- optimisation ---
    lr: float = 0.01
    lr_schedule: str = "const"  # velocity net only (recon_lr stays constant): 'const' | 'cosine'
    #                             (lr -> lr_min over the epochs after the warm-up)
    lr_warmup: int = 0  # velocity net: linear ramp lr/N, 2lr/N, ..., lr over the first N epochs
    lr_min: float = 0.0  # cosine floor (absolute lr)
    recon_lr: float = 0.1
    recon_eps: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 100
    learn_recon: bool = True
    interval: int = 0  # 0 -> epochs (alternation disabled)
    motion_warmup: int = 0
    use_nreps: bool = False  # F2: default off (only supported cmr mode)
    template_warp_recon: bool = False  # recon_t = warp(template, phi); only the motion net is learned
    fixed_motion: str = ""  # '' learn the motion | 'gt' the known motion of synth_flow frames (oracle)
    #                         | 'identity' (no motion) | path to a res.pt whose 'phi' is used as-is
    float64: bool = False    # run the whole pipeline in double precision (data + model + losses)

    # --- losses ---
    loss: str = "mse"  # F3: MSE for k-space data fidelity
    sim_domain: str = "fourier"  # 'fourier' (k-space DC), 'image' (image-space DC; equal to
    #                              fourier on fully-sampled data by Parseval, FFT unitary),
    #                              'fourier_mag' (|masked k-space| DC -- phase-insensitive), or
    #                              'image_mag' (|magnitude image| DC -- phase-insensitive)
    lambda_st: float = 1.0
    lambda_grad_phi: float = 1e-2  # pixel-unit (isotropic) ||Dphi - Id||, see losses.grad_phi_loss
    lambda_grd: float = 1.0
    lambda_lap: float = 1.0
    lambda_hel: float = 1.0
    lambda_recon: float = 1.0
    lambda_rl2: float = 1.0
    lambda_mcdc: float = 0.0
    lambda_detJ: float = 0.0
    lambda_logdetJ: float = 0.0
    imdiff_warp_mag: bool = False  # warp magnitude (not complex) in the E3 image loss
    joint_objective: str = "default"  # 'default': lambda_st * mean MSE + lambda_recon * mean_T reg(|x|)
    #                                   'init': the nmAPG init's energy, summed over frames --
    #                                   lambda_st * 0.5*sum|MFx - y|^2 + lambda_init_recon * reg
    #                                   (on |x| or Re/Im as init_reg_abs), see recon_init

    # --- derived / internal (filled by finalize) ---
    schedule: list = field(default_factory=lambda: [1])
    func_kwargs: dict = field(default_factory=dict)

    @classmethod
    def field_names(cls):
        return {f.name for f in fields(cls)}

    @classmethod
    def from_yaml(cls, path):
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        # Coerce to the annotated field type. Guards against the PyYAML gotcha where an
        # unsigned exponent like "1.0e4" is loaded as a str instead of a float.
        coerced = {k: _coerce_to_type(v, known[k].type) for k, v in data.items()}
        return cls(**coerced)

    @classmethod
    def from_saved(cls, data):
        """Rebuild the Config stored in a res.pt. Unlike from_dict, keys this version no longer
        knows (fields removed since the run, e.g. 'hard_dc') are dropped with a warning instead of
        raising, so old runs stay loadable."""
        known = cls.field_names()
        unknown = sorted(set(data) - known)
        if unknown:
            warnings.warn(f"Ignoring config keys unknown to this version: {unknown}")
        return cls.from_dict({k: v for k, v in data.items() if k in known})

    def finalize(self):
        """Normalise coupled fields once, instead of mutating config all over main.py."""
        if self.recon_scale == 0:
            self.recon_scale = None
        if self.step_size == 0:
            self.step_size = None
        if self.interval == 0:
            self.interval = self.epochs
        return self

    def to_dict(self):
        return asdict(self)


def build_velocity_kwargs(config: Config, dims: int) -> dict:
    """Build the velocity-network constructor kwargs for a given spatial dimensionality."""
    layers = [dims] + config.siren_depth * [config.siren_dim] + [dims]
    kwargs = {
        "layers": layers,
        "omega": config.siren_omega,
        "last_init_zero": config.last_init_zero,
        "last_init_std": config.last_init_std,
    }
    if "group" in config.func_name:
        kwargs["groups"] = config.time_points - 1
    return kwargs


def parse_cli(argv=None) -> Config:
    """Load a Config from ``--config file.yaml`` plus optional ``--key value`` overrides."""
    parser = argparse.ArgumentParser(description="space-time motion reconstruction")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="override config fields, e.g. --set epochs=5 lr=1e-4",
    )
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config) if args.config else Config()
    for override in args.set:
        key, _, raw = override.partition("=")
        if key not in Config.field_names():
            raise ValueError(f"Unknown override key: {key!r}")
        current = getattr(config, key)
        setattr(config, key, _coerce(raw, current))
    return config.finalize()


def _coerce_to_type(value, type_annotation):
    """Coerce a YAML-loaded value to the dataclass field's declared type."""
    ann = str(type_annotation)
    if value is None:
        # An explicit null (e.g. ``reg_alpha:`` in YAML) stays None regardless of the
        # declared type, so optional fields can be set back to their "keep pretrained"
        # sentinel without tripping float(None).
        return None
    if "bool" in ann:
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)
    if "int" in ann and "float" not in ann:
        return int(float(value))
    if "float" in ann:
        return float(value)
    return value


def _coerce(raw: str, current):
    """Coerce a string override to the type of the current field value."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if current is None:
        # Optional field (e.g. reg_alpha) whose current value is the None sentinel.
        # "none"/"null"/"" set it back to None; anything else is treated as a float.
        if raw.lower() in ("none", "null", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw
