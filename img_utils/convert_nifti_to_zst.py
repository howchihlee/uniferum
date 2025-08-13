import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from zst_utils import load_zst16, save_zst16


def convert_and_verify(input_file: str, output_file: str, verify: bool = True):
    img = nib.load(input_file).get_fdata().astype("int16")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_zst16(output_file, img)
    if verify:
        loaded_img = load_zst16(output_file)
        assert np.array_equal(img, loaded_img), "Arrays do not match!"


def main(batch_id, parquet_path, cache_file):
    df = pd.read_parquet(parquet_path)
    df = df[df.batch_id == batch_id]

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            complete = set(line.strip() for line in f)
    else:
        complete = set()
    os.makedirs(cache_dir, exist_ok=True)

    for inputs in zip(df.nii, df.zst):
        in_nii = inputs[0]
        if in_nii in complete:
            continue
        try:
            convert_and_verify(in_nii, inputs[1])
            with open(cache_file, "a") as f:
                f.write(f"{in_nii}\n")
            complete.add(in_nii)
        except Exception as e:
            print(f"Error processing {in_nii}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert nifti to zst on a batch of NIfTI files."
    )
    parser.add_argument("batch_id", type=int, help="Batch ID to process")
    parser.add_argument("input_filepath", type=str, help="Batch ID to process")
    parser.add_argument("project_id", type=str, help="project_id to save cached files")
    args = parser.parse_args()

    cache_dir = "./cache"
    cache_file = os.path.join(cache_dir, args.project_id, f"{args.batch_id}.txt")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    main(args.batch_id, args.input_filepath, cache_file)
