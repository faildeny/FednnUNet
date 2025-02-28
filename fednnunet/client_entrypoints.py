import argparse

import torch

from fednnunet.client import run_client


def client_entry():
    parser = argparse.ArgumentParser()

    # Additional arguments for the federated setup
    # parser.add_argument('task', type=str, help='Determines the task to be performed. Options are: extract_fingerprint, plan_and_preprocess or train')
    parser.add_argument(
        "--port", type=int, required=True, help="Port number of the server to listen on"
    )

    subparsers = parser.add_subparsers(
        help="Select the nnUNetv2 command to be executed", dest="task"
    )

    # Arguments for nnUNetv2_train command
    parser_train = subparsers.add_parser("train", help="Run nnUNetv2 training")

    parser_train.add_argument(
        "dataset_name_or_id", type=str, help="Dataset name or ID to train with"
    )
    parser_train.add_argument(
        "configuration", type=str, help="Configuration that should be trained"
    )
    parser_train.add_argument(
        "fold",
        type=str,
        help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.",
    )
    parser_train.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer",
    )
    parser_train.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans",
    )
    parser_train.add_argument(
        "-pretrained_weights",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only "
        "be used when actually training. Beta. Use with caution.",
    )
    parser_train.add_argument(
        "-num_gpus",
        type=int,
        default=1,
        required=False,
        help="Specify the number of GPUs to use for training",
    )
    parser_train.add_argument(
        "--use_compressed",
        default=False,
        action="store_true",
        required=False,
        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
        "data is much more CPU and (potentially) RAM intensive and should only be used if you "
        "know what you are doing",
    )
    parser_train.add_argument(
        "--npz",
        action="store_true",
        required=False,
        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
        "segmentations). Needed for finding the best ensemble.",
    )
    parser_train.add_argument(
        "--c",
        action="store_true",
        required=False,
        help="[OPTIONAL] Continue training from latest checkpoint",
    )
    parser_train.add_argument(
        "--val",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.",
    )
    parser_train.add_argument(
        "--val_best",
        action="store_true",
        required=False,
        help="[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead "
        "of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! "
        "WARNING: This will use the same 'validation' folder as the regular validation "
        "with no way of distinguishing the two!",
    )
    parser_train.add_argument(
        "--disable_checkpointing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
        "you dont want to flood your hard drive with checkpoints.",
    )
    parser_train.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )

    # Arguments for nnUNetv2_plan_and_preprocess command
    parser_plan_and_preprocess = subparsers.add_parser(
        "plan_and_preprocess", help="Run nnUNetv2 planning and preprocessing"
    )

    parser_plan_and_preprocess.add_argument(
        "-d",
        nargs="+",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser_plan_and_preprocess.add_argument(
        "-fpe",
        type=str,
        required=False,
        default="DatasetFingerprintExtractor",
        help="[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is "
        "'DatasetFingerprintExtractor'.",
    )
    parser_plan_and_preprocess.add_argument(
        "-npfp",
        type=int,
        default=8,
        required=False,
        help="[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8",
    )
    parser_plan_and_preprocess.add_argument(
        "--verify_dataset_integrity",
        required=False,
        default=False,
        action="store_true",
        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
        "each dataset!",
    )
    parser_plan_and_preprocess.add_argument(
        "--no_pp",
        default=False,
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no "
        "preprocesing). Useful for debugging.",
    )
    parser_plan_and_preprocess.add_argument(
        "--clean",
        required=False,
        default=False,
        action="store_true",
        help="[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a "
        "fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU "
        "CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!",
    )
    parser_plan_and_preprocess.add_argument(
        "-pl",
        type=str,
        default="ExperimentPlanner",
        required=False,
        help="[OPTIONAL] Name of the Experiment Planner class that should be used. Default is "
        "'ExperimentPlanner'. Note: There is no longer a distinction between 2d and 3d planner. "
        "It's an all in one solution now. Wuch. Such amazing.",
    )
    parser_plan_and_preprocess.add_argument(
        "-gpu_memory_target",
        default=None,
        type=float,
        required=False,
        help="[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target (in GB). Default: None (=Planner "
        "class default is used). Changing this will "
        "affect patch and batch size and will "
        "definitely affect your models performance! Only use this if you really know what you "
        "are doing and NEVER use this without running the default nnU-Net first as a baseline.",
    )
    parser_plan_and_preprocess.add_argument(
        "-preprocessor_name",
        default="DefaultPreprocessor",
        type=str,
        required=False,
        help="[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in "
        "nnunetv2.preprocessing. Default: 'DefaultPreprocessor'. Changing this may affect your "
        "models performance! Only use this if you really know what you "
        "are doing and NEVER use this without running the default nnU-Net first (as a baseline).",
    )
    parser_plan_and_preprocess.add_argument(
        "-overwrite_target_spacing",
        default=None,
        nargs="+",
        required=False,
        help="[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres "
        "configurations. Default: None [no changes]. Changing this will affect image size and "
        "potentially patch and batch "
        "size. This will definitely affect your models performance! Only use this if you really "
        "know what you are doing and NEVER use this without running the default nnU-Net first "
        "(as a baseline). Changing the target spacing for the other configurations is currently "
        "not implemented. New target spacing must be a list of three numbers!",
    )
    parser_plan_and_preprocess.add_argument(
        "-overwrite_plans_name",
        default=None,
        required=False,
        help="[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, "
        "-preprocessor_name or "
        "-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a "
        "differently named plans file such that the nnunet default plans are not "
        "overwritten. You will then need to specify your custom plans file with -p whenever "
        "running other nnunet commands (training, inference etc)",
    )
    parser_plan_and_preprocess.add_argument(
        "-c",
        required=False,
        default=["2d", "3d_fullres", "3d_lowres"],
        nargs="+",
        help="[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres "
        "3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data "
        "from 3d_fullres. Configurations that do not exist for some dataset will be skipped.",
    )
    parser_plan_and_preprocess.add_argument(
        "-np",
        type=int,
        nargs="+",
        default=None,
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with -c. If it's a "
        "list of numbers this list must have as many elements as there are configurations. We "
        "then iterate over zip(configs, num_processes) to determine then umber of processes "
        "used for each configuration. More processes is always faster (up to the number of "
        "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
        "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
        "often than not the number of processes that can be used is limited by the amount of "
        "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
        "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
        "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
    )
    parser_plan_and_preprocess.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )

    args = parser.parse_args()

    if args.task == "train":
        assert args.device in [
            "cpu",
            "cuda",
            "mps",
        ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
        if args.device == "cpu":
            # let's allow torch to use hella threads
            import multiprocessing

            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device("cpu")
        elif args.device == "cuda":
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device("cuda")
        else:
            device = torch.device("mps")
    else:
        args.dataset_name_or_id = args.d[0]

    run_client(args, device)


if __name__ == "__main__":
    client_entry()
