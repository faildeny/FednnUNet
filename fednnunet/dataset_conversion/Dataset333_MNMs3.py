import csv
import os
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.dataset_conversion.Dataset027_ACDC import make_out_dirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_preprocessed
from sklearn.model_selection import StratifiedKFold


def generate_MnM3(
    dataset_path: Path,
    csv_file: str,
    centre_id: int = None,
    save_splits=True,
    image_reader="NibabelIO",
):
    # Get patient ids from the csv file
    patient_ids = []
    patient_info = {}
    corrupted_samples = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        patient_index = headers.index("ID")
        centre_index = headers.index("Centre")
        for row in reader:
            patient_id = row[patient_index]
            patient_id = str(patient_id).zfill(3)
            patient_ids.append(patient_id)
            patient_info[patient_id] = {"centre": row[centre_index]}

            patient_dir = dataset_path / patient_id
            if not patient_dir.exists():
                print(f"Patient {patient_id} does not have a directory.")
                continue

            suffixes = ["ED", "ES", "ED_gt", "ES_gt"]
            corrupted_files = False
            for suffix in suffixes:
                if not (patient_dir / f"{patient_id}_{suffix}.nii.gz").exists():
                    if not (patient_dir / f"{patient_id}_SA_{suffix}.nii.gz").exists():
                        print(
                            f"Patient {patient_id} does not have a {suffix}.nii.gz file."
                        )
                    else:
                        patient_info[patient_id][suffix] = (
                            patient_dir / f"{patient_id}_SA_{suffix}.nii.gz"
                        )
                else:
                    patient_info[patient_id][suffix] = (
                        patient_dir / f"{patient_id}_{suffix}.nii.gz"
                    )

                # Check if files are readable
                if corrupted_files:
                    continue
                try:
                    if (
                        image_reader == "NibabelIO"
                    ):  # Used as deafult, because SimpleITKIO can't read some samples from the dataset
                        nib.load(patient_info[patient_id][suffix])
                    elif image_reader == "SimpleITKIO":
                        sitk.ReadImage(patient_info[patient_id][suffix])
                    else:
                        raise ValueError(f"Unsupported image reader: {image_reader}")
                except Exception as e:
                    corrupted_files = True
                    print(
                        f"File {patient_info[patient_id][suffix]} has unsupported format: {e} for {image_reader} reader."
                    )

            if corrupted_files:
                # Remove patient from dictionary
                patient_info.pop(patient_id)
                patient_ids.remove(patient_id)
                corrupted_samples.append(patient_id)

    if corrupted_samples:
        print(
            f"{len(corrupted_samples)} patients: {corrupted_samples} have unsupported file format."
        )

    # print("Done checking files.")

    # Perform stratified 5-fold cross validation based on the centres
    centre_folds = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    patient_ids = list(patient_info.keys())
    patient_centres = [patient_info[patient_id]["centre"] for patient_id in patient_ids]
    for fold, (train_index, test_index) in enumerate(
        skf.split(patient_ids, patient_centres)
    ):
        train_patients = [patient_ids[i] for i in train_index]
        test_patients = [patient_ids[i] for i in test_index]

        if centre_id is not None:
            train_patients = [
                patient_id
                for patient_id in train_patients
                if patient_info[patient_id]["centre"] == str(centre_id)
            ]
            test_patients = [
                patient_id
                for patient_id in test_patients
                if patient_info[patient_id]["centre"] == str(centre_id)
            ]

        train_file_names = [f"{patient_id}_ED" for patient_id in train_patients] + [
            f"{patient_id}_ES" for patient_id in train_patients
        ]
        test_file_names = [f"{patient_id}_ED" for patient_id in test_patients] + [
            f"{patient_id}_ES" for patient_id in test_patients
        ]
        split_dict = {"train": train_file_names, "val": test_file_names}

        centre_folds[fold] = {
            "train": train_patients,
            "test": test_patients,
            "split_dict": split_dict,
        }

    if centre_id:
        dataset_id = 300 + centre_id
    else:
        dataset_id = 300

    task_name = "MNMs3_fed" if centre_id else "MNMs3_centralized"
    out_dir, out_train_dir, out_labels_dir, out_test_dir = make_out_dirs(
        dataset_id, task_name=task_name
    )

    def save_subset(patient_ids, out_samples_dir):
        for patient_id in patient_ids:
            # Copy src files to out_dir
            for suffix in ["ED", "ES"]:
                sample_src = patient_info[patient_id][suffix]
                sample_dst = out_samples_dir / f"{patient_id}_{suffix}_0000.nii.gz"
                gt_src = patient_info[patient_id][f"{suffix}_gt"]
                gt_dst = out_labels_dir / f"{patient_id}_{suffix}.nii.gz"
                os.system(f"cp {sample_src} {sample_dst}")
                os.system(f"cp {gt_src} {gt_dst}")
            # print(f"Copied patient {patient_id}")

    save_subset(train_patients, out_train_dir)

    if save_splits:
        splits_list = [fold_split["split_dict"] for fold_split in centre_folds.values()]
        # Already create preprocessed directory for the dataset
        preprocessed_dir = (
            nnUNet_preprocessed.replace('"', "") + f"/Dataset{dataset_id}_{task_name}/"
        )
        os.makedirs(preprocessed_dir, exist_ok=True)
        save_json(splits_list, preprocessed_dir + "splits_final.json")
        print(
            f"Saved splits for dataset {dataset_id} in {preprocessed_dir}splits_final.json"
        )

        # No test set, only cross validation by nnunet
        save_subset(test_patients, out_train_dir)
        num_training_cases = (
            len(train_patients) + len(test_patients)
        ) * 2  # 2 since we have ED and ES for each patient

    else:
        save_subset(test_patients, out_test_dir)
        num_training_cases = (
            len(train_patients) * 2
        )  # 2 since we have ED and ES for each patient

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={"background": 0, "LVBP": 1, "LVM": 2, "RV": 3},
        file_ending=".nii.gz",
        overwrite_image_reader_writer=image_reader,  # change image reader as default SimpleITKIO can't read some samples from the dataset
        num_training_cases=num_training_cases,
    )

    print(
        f"Prepared dataset for centre {centre_id} with {len(train_patients) + len(test_patients)} patients"
    )


if __name__ == "__main__":
    import argparse

    class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=RawTextArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="MNMs conversion utility helper. This script can be used to convert MNMs data into the expected nnUNet "
        "format. It can also be used to create additional custom splits, for explicitly training on combinations "
        "of vendors A and B (see `--custom-splits`).\n"
        "If you wish to generate the custom splits, run the following pipeline:\n\n"
        "(1) Run `Dataset114_MNMs -i <raw_Data_dir>\n"
        "(2) Run `nnUNetv2_plan_and_preprocess -d 114 --verify_dataset_integrity`\n"
        "(3) Start training, but stop after initial splits are created: `nnUNetv2_train 114 2d 0`\n"
        "(4) Re-run `Dataset114_MNMs`, with `-s True`.\n"
        "(5) Re-run training.\n",
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="/data/MMs/",
        help="The downloaded MNMs dataset dir. Should contain a csv file, as well as Training, Validation and Testing "
        "folders.",
    )
    parser.add_argument(
        "-c",
        "--csv_file_name",
        type=str,
        default="211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        help="The csv file containing the dataset information.",
    ),
    parser.add_argument(
        "-d", "--dataset_id", type=int, default=114, help="nnUNet Dataset ID."
    )
    parser.add_argument(
        "-s",
        "--save_splits",
        type=bool,
        default=True,
        help="Save splits in nnUNet preprocessed directory.",
    )

    parser.add_argument(
        "--centre_id",
        nargs="+",
        type=int,
        default=None,
        help="Populate dataset for selected data centre. Accepts multiple values to create separate datasets. If not specified, the entire centralized dataset is populated. ",
    )

    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)
    if not args.centre_id:
        centre_ids = [None]
    else:
        centre_ids = set(args.centre_id)

    for centre_id in centre_ids:
        print(f"Populating dataset for centre {centre_id}")
        generate_MnM3(
            args.input_folder,
            args.csv_file_name,
            centre_id=centre_id,
            save_splits=args.save_splits,
        )

    print("Done!")
