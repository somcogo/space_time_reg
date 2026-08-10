"""SPARE CBCT loader (2D fan-beam mid-slice approximation).

Maps the SPARE challenge's raw cone-beam projections onto the same dense-forward-operator
+ mask contract the MRI pipeline uses (see `_cmr_load`/`_cmr_operators` in data_utils.py):
a fixed dense angle grid plays the role of "full k-space", and a per-frame boolean mask
selects which angles were actually measured in that respiratory-phase bin. Only the
central detector row of each raw projection is used, approximating the true 3D cone-beam
acquisition as a 2D fan-beam sinogram of one axial slice.

Only `cbct_P<patient>T<scan>` (e.g. `cbct_P1T01`) datasets are supported -- i.e. Elekta
"T" (test) scans, which have both a sparse `Proj/` acquisition (the `raw`/undersampled
measurement) and a much denser `Proj_Full/` acquisition of the same patient/scan used to
build a genuine per-phase-bin reference (`gt`). "V" (validation) scans only have `Proj/`
plus a pre-computed classical FDK reconstruction (`FDKRecon/`), not a comparable raw
reference, so they aren't wired in here.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

from .ct_utils import FanBeamBackprojector, FanBeamFBP, FanBeamProjector

_SPARE_ROOT = "data/raw/spare/ClinicalElektaDatasets"
_DATASET_RE = re.compile(r"^cbct_P(\d+)T(\d+)$")


def _scan_dir(patient: str, scan_num: str) -> str:
    return os.path.join(_SPARE_ROOT, f"P{patient}", f"CE_P{patient}_T_{scan_num}")


def _parse_dataset_id(dataset: str) -> tuple[str, str]:
    m = _DATASET_RE.match(dataset)
    if not m:
        raise ValueError(f"Unrecognized cbct dataset id {dataset!r}; expected "
                         f"'cbct_P<patient>T<scan>', e.g. 'cbct_P1T01'")
    return m.group(1), m.group(2)


def _parse_geometry(proj_dir: str) -> tuple[np.ndarray, float, float]:
    """Return (gantry_angles_deg [N], source_to_isocenter_mm, source_to_detector_mm),
    ordered to match Proj_NNNNN.bin file order (RTKThreeDCircularGeometry XML)."""
    root = ET.parse(os.path.join(proj_dir, "Geometry.xml")).getroot()
    src_iso = float(root.find("SourceToIsocenterDistance").text)
    src_det = float(root.find("SourceToDetectorDistance").text)
    angles = np.array([float(p.find("GantryAngle").text) for p in root.findall(".//Projection")])
    return angles, src_iso, src_det


def _parse_resp_bin(proj_dir: str) -> np.ndarray:
    with open(os.path.join(proj_dir, "RespBin.csv")) as f:
        return np.array([int(x) for x in f.read().splitlines()])


def _load_detector_row(proj_dir: str, n_proj: int, row: int, det_size: int = 512) -> torch.Tensor:
    """Read Proj_00001.bin.. in file order, keep only detector row `row` of each.
    Returns [n_proj, det_size] float32 -- one row per raw projection."""
    rows = np.empty((n_proj, det_size), dtype=np.float32)
    for i in range(n_proj):
        path = os.path.join(proj_dir, f"Proj_{i + 1:05d}.bin")
        img = np.fromfile(path, dtype=np.float32).reshape(det_size, det_size)
        rows[i] = img[row]
    return torch.from_numpy(rows)


def _dense_grid(angles_deg: np.ndarray, resp_bin: np.ndarray, sino_rows: torch.Tensor,
                n_bins: int, det_size: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Scatter [n_proj, det_size] rows onto a [n_bins, n_angles, det_size] dense-angle-grid
    sinogram (angle-sorted so every bin shares the same fixed angle axis) plus the boolean
    mask marking which (bin, angle) entries were actually measured."""
    order = np.argsort(angles_deg)
    dense_angles = angles_deg[order]
    n_angles = len(dense_angles)
    sorted_bin = resp_bin[order]
    sorted_rows = sino_rows[order]

    dense_sino = torch.zeros(n_bins, n_angles, det_size)
    mask = torch.zeros(n_bins, n_angles, det_size, dtype=torch.bool)
    for bin_idx in range(n_bins):
        selected = sorted_bin == (bin_idx + 1)
        dense_sino[bin_idx, selected] = sorted_rows[selected]
        mask[bin_idx, selected] = True
    return dense_angles, dense_sino, mask


def _detector_spacing_at_isocenter(src_iso: float, src_det: float,
                                   raw_det_spacing_mm: float = 0.8) -> float:
    # Magnify the physical detector pixel spacing back down to the isocenter plane, so the
    # reconstructed image's pixel spacing matches the 1mm FDKRecon volumes' convention.
    return raw_det_spacing_mm * (src_iso / src_det)


def _geometry_for_config(config: argparse.Namespace) -> tuple[str, float, float, float, np.ndarray]:
    """Everything get_operators needs to rebuild the same forward operator get_data used,
    derived purely from config (so the two functions stay independently callable, matching
    the cmr family's contract)."""
    patient, scan_num = _parse_dataset_id(config.dataset)
    raw_dir = os.path.join(_scan_dir(patient, scan_num), "Proj")
    angles_deg, src_iso, src_det = _parse_geometry(raw_dir)
    det_spacing = _detector_spacing_at_isocenter(src_iso, src_det)
    dense_angles = np.sort(angles_deg)
    return raw_dir, src_iso, src_det, det_spacing, dense_angles


def get_data(config: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patient, scan_num = _parse_dataset_id(config.dataset)
    scan_dir = _scan_dir(patient, scan_num)
    raw_dir = os.path.join(scan_dir, "Proj")
    full_dir = os.path.join(scan_dir, "Proj_Full")

    det_size = 512
    row = det_size // 2 + config.slice_number
    n_bins = config.time_points

    raw_angles, src_iso, src_det = _parse_geometry(raw_dir)
    raw_rows = _load_detector_row(raw_dir, len(raw_angles), row, det_size)
    raw_bin = _parse_resp_bin(raw_dir)
    raw_dense_angles, raw_dense_sino, mask = _dense_grid(raw_angles, raw_bin, raw_rows,
                                                          n_bins, det_size)

    full_angles, _, _ = _parse_geometry(full_dir)
    full_rows = _load_detector_row(full_dir, len(full_angles), row, det_size)
    full_bin = _parse_resp_bin(full_dir)
    full_dense_angles, full_dense_sino, _ = _dense_grid(full_angles, full_bin, full_rows,
                                                         n_bins, det_size)

    det_spacing = _detector_spacing_at_isocenter(src_iso, src_det)
    det_dist = src_det - src_iso

    # gt: reconstruct each phase bin from the denser Proj_Full subset (its own, slightly
    # different angle sampling), then re-forward-project through the *raw* scan's dense
    # angle grid so `gt` and `raw` live in the same sinogram-domain shape -- mirrors
    # `gt_im = full_adj(gt_kspace_data)` followed by `forw_subs(gt_im)` in the MRI pipeline.
    device = config.device
    full_fbp = FanBeamFBP(det_size, np.deg2rad(full_dense_angles), src_iso, det_dist,
                          det_spacing).to(device)
    gt_img = full_fbp(full_dense_sino.unsqueeze(1).to(device))

    raw_forw = FanBeamProjector(det_size, np.deg2rad(raw_dense_angles), src_iso, det_dist,
                                det_spacing).to(device)
    gt_sino = raw_forw(gt_img).detach().cpu()

    raw_sino = raw_dense_sino.unsqueeze(1)
    mask = mask.unsqueeze(1)

    # torch-radon's ramp-filter normalization convention isn't calibrated against this
    # scanner's raw intensity units (a generic Radon-transform library has no way to know
    # them), so gt_sino/gt_img land at the wrong absolute scale. Fix with a single
    # least-squares scale factor fit on the actually-measured (masked) raw entries --
    # standard practice when bolting a generic projector onto real scanner data with an
    # unknown absolute calibration constant.
    measured_gt = gt_sino[mask]
    measured_raw = raw_sino[mask]
    denom = measured_gt.pow(2).sum()
    alpha = (measured_raw * measured_gt).sum() / denom if denom > 0 else torch.tensor(1.0)
    gt_sino = gt_sino * alpha

    # Rescale to unit-ish magnitude (matching the pre-normalized "_norm" convention the
    # cmr family's .pt files already use): raw's own measured magnitude in physical units
    # (line-integral values ~O(1-6)) is much larger than what the velocity-net/regularizer
    # learning rates and lambda weights in soft_con-style configs are tuned for, and caused
    # the init-recon gradient descent to diverge to NaN before this was added.
    norm_scale = measured_raw.abs().mean().clamp_min(1e-6)
    raw_sino = raw_sino / norm_scale
    gt_sino = gt_sino / norm_scale
    return raw_sino, gt_sino, mask


def get_operators(config: argparse.Namespace, mask: torch.Tensor) -> tuple[nn.Module, ...]:
    _, src_iso, src_det, det_spacing, dense_angles = _geometry_for_config(config)
    det_size = 512
    det_dist = src_det - src_iso
    angles_rad = np.deg2rad(dense_angles)

    full_forw = FanBeamProjector(det_size, angles_rad, src_iso, det_dist, det_spacing)
    full_adj = FanBeamFBP(det_size, angles_rad, src_iso, det_dist, det_spacing)
    forw_subs = FanBeamProjector(det_size, angles_rad, src_iso, det_dist, det_spacing, mask=mask)
    forw_subs_adj = FanBeamBackprojector(det_size, angles_rad, src_iso, det_dist, det_spacing,
                                         mask=mask)
    return full_forw, full_adj, forw_subs, forw_subs_adj
