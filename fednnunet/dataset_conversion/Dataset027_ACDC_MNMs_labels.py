import json
import os
import shutil
from pathlib import Path

import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    save_json,
)
from nnunetv2.dataset_conversion.Dataset027_ACDC import create_ACDC_split, make_out_dirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw


# Slight modification to the original ACDC conversion script that copies the test gt files as well
def copy_files(
    src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path
):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted(
        [f for f in (src_data_folder / "training").iterdir() if f.is_dir()]
    )
    patients_test = sorted(
        [f for f in (src_data_folder / "testing").iterdir() if f.is_dir()]
    )

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if (
                file.suffix == ".gz"
                and "_gt" not in file.name
                and "_4d" not in file.name
            ):
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, train_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    # Copy test files.
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if (
                file.suffix == ".gz"
                and "_gt" not in file.name
                and "_4d" not in file.name
            ):
                shutil.copy(file, test_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    return num_training_cases


def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(
        Path(src_data_folder), train_dir, labels_dir, test_dir
    )

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


# Read gt files from input directory and remap labels
def convert_labels(input_dir, output_dir, labels_remapping):

    # Copy input dir to output
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

    input_labels_dir = os.path.join(input_dir, "labelsTr")
    output_labels_dir = os.path.join(output_dir, "labelsTr")

    for gt_image_name in os.listdir(input_labels_dir):
        if not gt_image_name.endswith(".nii.gz"):
            continue
        gt_image = nib.load(os.path.join(input_labels_dir, gt_image_name))
        gt_data = gt_image.get_fdata()

        remmaped_data = gt_data.copy()
        # Remap labels
        for old_label, new_label in labels_remapping.items():
            remmaped_data[gt_data == old_label] = new_label

        gt_image = nib.Nifti1Image(remmaped_data, gt_image.affine)
        nib.save(gt_image, os.path.join(output_labels_dir, gt_image_name))

    # Open dataset.json and remap labels
    dataset_json_path = os.path.join(output_dir, "dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
        for label, new_id in zip(dataset_json["labels"], labels_remapping):
            dataset_json["labels"][label] = labels_remapping[new_id]

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("Labels remapped successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded ACDC dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d",
        "--dataset_id",
        required=False,
        type=int,
        default=27,
        help="nnU-Net Dataset ID, default: 27",
    )
    args = parser.parse_args()
    print("Converting...")
    convert_acdc(args.input_folder, args.dataset_id)

    dataset_name = f"Dataset{args.dataset_id:03d}_{'ACDC'}"
    labelsTr = join(nnUNet_raw, dataset_name, "labelsTr")
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(preprocessed_folder)
    split = create_ACDC_split(labelsTr)
    save_json(split, join(preprocessed_folder, "splits_final.json"), sort_keys=False)

    print("Converting labels...")

    # Remap labels to MNMs format
    labels_remapping = {
        0: 0,
        1: 3,
        2: 2,
        3: 1,
    }

    convert_labels(
        join(nnUNet_raw, dataset_name),
        join(nnUNet_raw, dataset_name + "_MNMs"),
        labels_remapping,
    )

    print("Done!")
