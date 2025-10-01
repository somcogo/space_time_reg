# preprocess_oasis.py
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

from thomas_registration.spatial import spatial, resample

sitk.ProcessObject.SetGlobalWarningDisplay(False)


# --- CONFIG ---
OASIS_ROOT = Path("data/oasis-1/oasis_cross-sectional_disc1/disc1")          # folder containing OASIS files (nifti/Analyze)
OUT_DIR = Path("data/oasis-1/preproc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to an MNI152 template (NIfTI). If you have Nilearn/Nibabel installed you can
# download or point to a local copy of MNI152_T1_1mm (commonly available).
MNI_TEMPLATE = Path("data/MNI152_T1_1mm_brain.nii.gz")

# Final crop size (paper): (D, H, W) = (162, 192, 144) — confirm order below
TARGET_SHAPE = (162, 192, 144)  # note: we'll map to (Z,Y,X) as nibabel expects (X,Y,Z) ordering

# --- HELPERS ---
def sitk_read(path):
    return sitk.ReadImage(str(path))

def sitk_to_nib(sitk_img):
    arr = sitk.GetArrayFromImage(sitk_img)  # SITK: z,y,x -> numpy: (z,y,x)
    affine = np.eye(4)
    # build affine from SITK spacing/origin/direction (approximate)
    spacing = np.array(sitk_img.GetSpacing())   # (x,y,z)
    origin = np.array(sitk_img.GetOrigin())     # (x,y,z)
    direction = np.array(sitk_img.GetDirection()).reshape(3,3)
    # nibabel expects affine mapping (i,j,k)->world; compose direction*diag(spacing)
    affine[:3,:3] = direction @ np.diag(spacing)
    affine[:3,3] = origin
    # nib expects data in (x,y,z) order so we must transpose
    data = np.transpose(arr, (2,1,0))  # (z,y,x) -> (x,y,z)
    return nib.Nifti1Image(data, affine)

def center_crop_or_pad(img_nib, target_shape):
    # img_nib data shape is (X,Y,Z) -> convert to (Z,Y,X) to think in usual medical (D,H,W)
    data = img_nib.get_fdata()
    data_zyx = np.transpose(data, (2,1,0))  # (Z,Y,X)
    D_t, H_t, W_t = target_shape
    D, H, W = data_zyx.shape
    # compute cropping/padding indices centered
    start = [max(0, (s - t)//2) for s,t in zip((D, H, W), (D_t, H_t, W_t))]
    end = [start[i] + [D_t, H_t, W_t][i] for i in range(3)]
    # create output with zeros
    out = np.zeros((D_t, H_t, W_t), dtype=data_zyx.dtype)
    # compute source slice ranges
    src_slices = [
        slice(start[i], min(end[i], (D, H, W)[i]))
        for i in range(3)
    ]
    # compute dest start (in case target bigger)
    dest_start = [max(0, -((s - t)//2)) if (s < t) else 0 for s,t in zip((D, H, W), (D_t, H_t, W_t))]
    dest_slices = [slice(dest_start[i], dest_start[i] + (src_slices[i].stop - src_slices[i].start)) for i in range(3)]
    out[dest_slices[0], dest_slices[1], dest_slices[2]] = data_zyx[src_slices[0], src_slices[1], src_slices[2]]
    # transpose back to (X,Y,Z)
    out_xyz = np.transpose(out, (2,1,0))
    # new affine: keep original affine but adjust origin so crop is centered correctly in world coords.
    new_affine = np.copy(img_nib.affine)
    # compute shift in world coords: shift = voxel_offset * voxel_size in brain axes (approx)
    voxel_sizes = np.sqrt((img_nib.affine[:3,:3] ** 2).sum(axis=0))
    shift_vox = np.array([start[2], start[1], start[0]])  # because of earlier transpositions
    new_affine[:3,3] = img_nib.affine[:3,3] + img_nib.affine[:3,:3] @ shift_vox
    return nib.Nifti1Image(out_xyz, new_affine)

# --- AFFINE REGISTRATION (SITK) ---
def affine_register_to_mni(moving_path, mni_path):
    # fixed = sitk_read(mni_path)
    # moving = sitk_read(moving_path)
    
    # print(np.array(fixed.GetDirection()).reshape(3,3), np.array(fixed.GetSpacing()), np.array(fixed.GetOrigin()), '\n',
    #       np.array(moving.GetDirection()).reshape(3,3), np.array(moving.GetSpacing()), np.array(moving.GetOrigin()))
    # print(sitk.GetArrayFromImage(fixed).shape, sitk.GetArrayFromImage(moving).shape)
    # transform = spatial(fixed=fixed, moving=moving, transform='affine')
    # resampled = resample(moving=moving, fixed=fixed, transform=transform)
    # print(np.array(resampled.GetDirection()).reshape(3,3), np.array(resampled.GetSpacing()), np.array(resampled.GetOrigin()), '\n',)

    # return resampled

    fixed = sitk_read(mni_path)
    moving = sitk_read(moving_path)
    # print(np.array(fixed.GetDirection()).reshape(3,3), np.array(fixed.GetSpacing()), np.array(fixed.GetOrigin()), '\n',
    #       np.array(moving.GetDirection()).reshape(3,3), np.array(moving.GetSpacing()), np.array(moving.GetOrigin()))
    # print(sitk.GetArrayFromImage(fixed).shape, sitk.GetArrayFromImage(moving).shape)
    # Setup registration
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    # multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    out_tx = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    # Resample moving onto fixed grid using the computed transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(out_tx)
    resampled = resampler.Execute(moving)
    # print(np.array(resampled.GetDirection()).reshape(3,3), np.array(resampled.GetSpacing()), np.array(resampled.GetOrigin()), '\n',
    #       sitk.GetArrayFromImage(resampled).shape)
    return resampled

# --- MAIN batch preprocessing ---
def preprocess_single(nifti_path, out_dir, mni_template, skull_stripped=False):
    # 1) If input not nifti, assume nifti or analyze already: use SimpleITK for registration
    try:
        resampled_sitk = affine_register_to_mni(nifti_path, mni_template)
    except Exception as e:
        print(f"REG FAIL for {nifti_path}: {e}")
        return None
    # moving = sitk_read(nifti_path)
    # 2) convert to nibabel
    # nib_img = sitk_to_nib(moving)
    nib_img = sitk_to_nib(resampled_sitk)
    # 3) optional: skull-strip step (not implemented here)
    if not skull_stripped:
        # If you need skull stripping, run an external tool (recommend HD-BET or FSL BET) and reload as nib_img
        pass
    # 4) crop / pad to target shape (paper uses (162,192,144))
    cropped = center_crop_or_pad(nib_img, TARGET_SHAPE)
    # cropped = nib_img
    # 5) normalize by max intensity (paper)
    data = cropped.get_fdata(dtype=np.float32)
    vmax = data.max() if data.max() != 0 else 1.0
    data_norm = (data / vmax).astype(np.float32)
    out_affine = cropped.affine
    # out_affine[1, 3] = out_affine[1, 3]
    # print(out_affine[1, 3], out_affine[3, 1])
    out_img = nib.Nifti1Image(data_norm, out_affine)
    # save
    out_name = out_dir / (Path(nifti_path).stem + "_mni_crop_norm.nii.gz")
    nib.save(out_img, str(out_name))
    return out_name

def batch_preprocess(oasis_root, out_dir, mni_template):
    """
    Preprocess OASIS-1 files matching:
    OAS1_xxxx_MR?/PROCESSED/MPRAGE/T88_111/OAS1_xxxx_MR?_mpr_n4_anon_111_t88_masked_gfc.*
    """
    # Regex for the expected path pattern
    pattern = re.compile(
        r"OAS1_\d{4}_MR\d/PROCESSED/MPRAGE/T88_111/"
        r"OAS1_\d{4}_MR\d_mpr_n4_anon_111_t88_masked_gfc\..*"
    )

    files = []
    for p in Path(oasis_root).rglob("*"):
        if p.is_file():
            rel_path = str(p.relative_to(oasis_root))
            if pattern.fullmatch(rel_path):
                files.append(p)

    results = []
    for f in tqdm(files):
        try:
            out = preprocess_single(str(f), out_dir, mni_template, skull_stripped=True)  
            if out:
                results.append((str(f), str(out)))
        except Exception as e:
            print("Error:", f, e)

    # Save mapping
    with open(out_dir / "mapping.csv", "w") as fh:
        fh.write("input,preprocessed\n")
        for i, o in results:
            fh.write(f"{i},{o}\n")

    print("Done. Preprocessed:", len(results))
    return results

if __name__ == "__main__":
    batch_preprocess(OASIS_ROOT, OUT_DIR, MNI_TEMPLATE)
