import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_img
from nilearn.image.image import _crop_img_to
from skimage import measure
from skimage.measure import label, regionprops


def apply_soft_tissue_window_level(data):
    window_center = 50
    window_width = 400
    return apply_window_level(data, window_center, window_width)


def apply_window_level(data, window_center, window_width):
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    window_leveled_data = np.clip(data, lower_bound, upper_bound)
    # Normalize to the range 0 to 1 (optional, can adjust based on output
    # requirement)
    window_leveled_data = (window_leveled_data - lower_bound) / window_width
    return window_leveled_data


def crop_nd_array_from_bbox(array, bbox):
    """
    Crops an N-dimensional array using the provided bounding box.

    Parameters:
    - array: np.ndarray, the input N-dimensional array.
    - bbox: tuple of tuples (each containing min and max indices for each dimension).

    Returns:
    - Cropped array.
    """
    slices = tuple(slice(min_idx, max_idx) for min_idx, max_idx in bbox)
    return array[slices]


def get_slicer_from_bbox(bbox):
    return tuple(slice(min_idx, max_idx) for min_idx, max_idx in bbox)


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largestCC


def resample_nifti_nilearn(
    nii_img, target_spacing=(1, 1, 1), interpolation="continuous"
):
    new_affine = nii_img.affine.copy()
    scalings = [b / a for a, b in zip(nii_img.header.get_zooms(), target_spacing)]
    new_affine[:3, :3] = new_affine[:3, :3] @ np.diag(scalings)

    # Resample the image
    resampled = resample_img(
        nii_img,
        target_affine=new_affine,
        target_shape=None,
        interpolation=interpolation,
    )
    return resampled


def get_centered_bbox(mask):
    """
    Given a 3D binary mask, find the bounding box, centroid, and a new bounding box centered
    at the centroid that includes the original bounding box.

    Parameters:
        mask (np.ndarray): A 3D binary numpy array (mask).

    Returns:
        tuple: (original_bbox, centroid, new_bbox) where
            - original_bbox: original bounding box of the mask (min, max) for each dimension.
            - centroid: centroid of the mask (x, y, z).
            - new_bbox: new bounding box centered at the centroid, which includes the original bbox.
    """
    labeled_mask = measure.label(mask)
    props = regionprops(labeled_mask)
    region = props[0]

    # Original Bounding Box (min, max) coordinates for each dimension
    # (min_row, min_col, min_depth, max_row, max_col, max_depth)
    bbox = region.bbox
    # Centroid (x, y, z) coordinates
    centroid = [round(c) for c in region.centroid]
    ndim = len(centroid)
    width = np.array(
        [max([c - x0, x1 - c]) for c, x0, x1 in zip(centroid, bbox[:ndim], bbox[ndim:])]
    )

    new_min = np.maximum(np.array(centroid) - width, 0)
    new_max = np.minimum(np.array(centroid) + width, mask.shape)

    # new_bbox = tuple(np.concatenate([new_min, new_max]))
    new_bbox = list(zip(new_min, new_max))

    return mask, new_bbox


def select_spacing(s):
    return min(max(round(s), 1), 5)


def crop_resize_clip(
    img_nii, slices, target_spacing=None, interpolation="continuous", clip=None
):

    cropped_nii = _crop_img_to(img_nii, slices)
    if target_spacing is not None:
        cropped_nii = resample_nifti_nilearn(
            cropped_nii, target_spacing=target_spacing, interpolation=interpolation
        )

    if clip is not None:
        data = cropped_nii.get_fdata().astype(np.int16)
        data = np.clip(data, -1000, 1000)
        cropped_nii = nib.Nifti1Image(data, cropped_nii.affine, cropped_nii.header)
        cropped_nii.set_data_dtype(np.int16)

    return cropped_nii


def _proc_img(img: np.array):
    st_img = (apply_soft_tissue_window_level(img) * 255).astype("uint8")
    LCCmask = getLargestCC(st_img > 0)
    LCCmask, bbox = get_centered_bbox(LCCmask)
    slices = get_slicer_from_bbox(bbox)
    return LCCmask, bbox, slices


def process_nii2npz_with_mask(inputs):
    if len(inputs) == 4:
        input_path, mask_path, output_img_path, output_mask_path = inputs
    elif len(inputs) == 2:
        input_path, output_img_path = inputs
        mask_path, output_mask_path = None, None
    else:
        return
    try:
        nii_img = nib.load(input_path)

        img = nii_img.get_fdata()
        LCCmask, bbox, slices = _proc_img(img)
        if mask_path is not None:
            nii_mask = nib.load(mask_path)
            seg = nii_mask.get_fdata().astype("int")
            lung_slices = np.where(
                np.isin(seg, np.array([10, 11, 12, 13, 14])).max(axis=(0, 1))
            )
            zmax, zmin = np.max(lung_slices), np.min(lung_slices)
            zslice = slice(
                max(slices[2].start, zmin), min(slices[2].stop, zmax), slices[2].step
            )
            slices = (slices[0], slices[1], zslice)
            cropped_mask = crop_resize_clip(
                nii_mask, slices, (1, 1, 1), interpolation="nearest"
            )
            cropped_mask = nib.as_closest_canonical(cropped_mask)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            nib.save(cropped_mask, output_mask_path)

        cropped_img = crop_resize_clip(nii_img, slices, (1, 1, 1), clip=True)
        cropped_img = nib.as_closest_canonical(cropped_img)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        nib.save(cropped_img, output_img_path)

    except:
        print(f"failed {input_path}")
        return


def main(batch_id, parquet_path, cache_file):
    df = pd.read_parquet(parquet_path)
    df = df[df.batch_id == batch_id]

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            complete = set(line.strip() for line in f)
    else:
        complete = set()
    os.makedirs(cache_dir, exist_ok=True)

    for inputs in zip(
        df.filepath_img, df.filepath_mask, df.output_img_path, df.output_mask_path
    ):
        in_nii = inputs[0]
        if in_nii in complete:
            continue
        try:
            process_nii2npz_with_mask(inputs)
            with open(cache_file, "a") as f:
                f.write(f"{in_nii}\n")
            complete.add(in_nii)
        except Exception as e:
            print(f"Error processing {in_nii}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize nii."
    )
    parser.add_argument("batch_id", type=int, help="Batch ID to process")
    parser.add_argument("input_filepath", type=str, help="Batch ID to process")
    parser.add_argument("project_id", type=str, help="project_id to save cached files")
    args = parser.parse_args()

    cache_dir = "./cache"
    cache_file = os.path.join(cache_dir, args.project_id, f"{args.batch_id}.txt")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    main(args.batch_id, args.input_filepath, cache_file)
