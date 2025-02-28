import argparse
import logging
import os
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Type, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Code, EvaluateRes, FitRes, GetParametersRes, Parameters, Status

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nnUNet"))
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiments,
    preprocess,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import (
    convert_dataset_name_to_id,
    maybe_convert_to_dataset_name,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from fednnunet.run_training import run_training

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


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


def state_dict_to_bytes(state_dict) -> bytes:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict) -> Parameters:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    tensors = state_dict_to_bytes(state_dict)
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> dict:
    """Converts bytes back to a PyTorch state_dict."""
    bytes_io = BytesIO(bytes_data)
    return torch.load(bytes_io)


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


class FlowerClient(fl.client.Client):

    def __init__(
        self,
        task: str = "train",
        args: argparse.Namespace = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.task = task
        self.dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
        self.dataset_id = convert_dataset_name_to_id(self.dataset_name)
        self.num_samples = None
        self.extract_fingerprint = False
        self.plan_experiment = False
        self.preprocess_dataset = False

        self.train = False
        if self.task == "train":
            self.train = True
            # this calls run_training but is not running any training, I did not change the name of the method for compatibility with regular nnUnet.
            self.trainer = run_training(
                args.dataset_name_or_id,
                args.configuration,
                args.fold,
                args.tr,
                args.p,
                args.pretrained_weights,
                args.num_gpus,
                args.use_compressed,
                args.npz,
                args.c,
                args.val,
                args.disable_checkpointing,
                args.val_best,
                device=device,
                return_trainer=True,
            )

            self.trainer.initialize()
            self.model = self.trainer.network
            self.trainer.on_train_start()

        if self.task == "plan_and_preprocess":
            self.extract_fingerprint = True
            self.plan_experiment = True
            self.preprocess_dataset = True
            self.gpu_memory_target_in_gb = args.gpu_memory_target

            if args.np is None:
                default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
                args.np = [
                    default_np[c] if c in default_np.keys() else 4 for c in args.c
                ]
            else:
                args.np = args.np
            if args.no_pp:
                self.preprocess_dataset = False

        if self.task == "extract_fingerprint" or self.extract_fingerprint:
            self.extract_fingerprint = True
            self.fingerprint = None
            self.local_fingerprint = None

        self.preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)

    def get_overlapping_keys(self, state_dict1, state_dict2):
        """Find keys that are present in both state_dicts and have the same shape."""
        overlapping_keys = set(state_dict1.keys()).intersection(state_dict2.keys())
        compatible_keys = [
            key
            for key in overlapping_keys
            if state_dict1[key].shape == state_dict2[key].shape
        ]
        return compatible_keys

    def replace_overlapping_keys(self, target_state_dict, source_state_dict):
        """Replace keys in the target_state_dict with the values from source_state_dict for overlapping keys."""
        overlapping_keys = self.get_overlapping_keys(
            target_state_dict, source_state_dict
        )

        for key in overlapping_keys:
            target_state_dict[key] = source_state_dict[key]

        return target_state_dict

    def get_fingerprint(self):
        if not self.local_fingerprint:
            # self.local_fingerprint = extract_fingerprint_dataset(self.dataset_id, clean=True)
            fingerprint_extractor_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "experiment_planning"),
                args.fpe,
                current_module="nnunetv2.experiment_planning",
            )
            self.local_fingerprint = extract_fingerprint_dataset(
                self.dataset_id,
                fingerprint_extractor_class=fingerprint_extractor_class,
                num_processes=args.npfp,
                check_dataset_integrity=args.verify_dataset_integrity,
                clean=True,
                verbose=args.verbose,
            )
            self.num_samples = len(self.local_fingerprint["shapes_after_crop"])
            save_json(
                self.local_fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint_local.json"),
            )
            self.fingerprint = self.local_fingerprint

        return self.fingerprint

    def get_parameters(self, fi):
        if self.extract_fingerprint:
            parameters = self.get_fingerprint()
            # print(f'Fingerprint with mean: {self.fingerprint["median_relative_size_after_cropping"]}')
        else:
            parameters = self.model.state_dict()

        parm = GetParametersRes(
            parameters=state_dict_to_parameters(parameters),
            status=Status(code=Code(0), message="caguento"),
        )
        return parm

    def set_parameters(self, parameters):

        common_state_dict = parameters_to_state_dict(parameters)

        if self.extract_fingerprint:
            self.fingerprint = common_state_dict
        else:
            # torch.save(common_state_dict,os.path.join(os.path.dirname(os.path.abspath(__file__)),'common_state_dict.arch'))
            # torch.save(self.model.state_dict(),os.path.join(os.path.dirname(os.path.abspath(__file__)),'local_state_dict.arch'))

            onset_subset_keys_dict = self.replace_overlapping_keys(
                self.model.state_dict(), common_state_dict
            )
            self.model.load_state_dict(onset_subset_keys_dict, strict=True)

    def fit(self, fi):
        self.set_parameters(fi.parameters)

        if self.extract_fingerprint:
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Fingerprint extracted"),
                num_examples=0,
                metrics={},
            )
        else:
            # adding try catch errors
            try:
                self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                raise

            tl = np.round(
                self.trainer.logger.my_fantastic_logging["train_losses"][-1], decimals=4
            )
            fr = FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message=""),
                num_examples=len(self.trainer.dataloader_train.generator._data),
                metrics={"loss": float(tl)},
            )
            return fr

    def evaluate(self, ei):
        # We need to update to the aggregated parameters, otherwise the model will be evaluated on local weights
        self.set_parameters(ei.parameters)

        if self.extract_fingerprint:
            save_json(
                self.fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint.json"),
            )
            logging.info(
                f"Federated dataset fingerprint saved to {join(self.preprocessed_output_folder, 'dataset_fingerprint.json')}"
            )
            if self.plan_experiment:
                plans_identifier = plan_experiments(
                    [self.dataset_id],
                    experiment_planner_class_name=args.pl,
                    gpu_memory_target_in_gb=args.gpu_memory_target,
                    preprocess_class_name=args.preprocessor_name,
                    overwrite_target_spacing=args.overwrite_target_spacing,
                    overwrite_plans_name=args.overwrite_plans_name,
                )
                logging.info(f"Experiment plan created for {self.dataset_name}")
                if self.preprocess_dataset:
                    preprocess(
                        [self.dataset_id],
                        plans_identifier=plans_identifier,
                        configurations=args.c,
                        num_processes=args.np,
                        verbose=args.verbose,
                    )
                    logging.info(f"Dataset {self.dataset_name} preprocessed")

            return EvaluateRes(
                status=Status(code=Code(0), message="Federated fingerprint saved"),
                loss=0.0,
                num_examples=1,
                metrics={},
            )

        vl = np.round(
            self.trainer.logger.my_fantastic_logging["val_losses"][-1], decimals=4
        )
        dc = [
            np.round(i, decimals=4)
            for i in self.trainer.logger.my_fantastic_logging[
                "dice_per_class_or_region"
            ][-1]
        ]

        er = EvaluateRes(
            status=Status(code=Code(0), message="yacasi"),
            loss=float(vl),
            num_examples=len(self.trainer.dataloader_val.generator._data),
            metrics={"fg_dice": float(np.nanmean(dc))},
        )

        return er


def run_client(args, device):

    # Initialize the client and pass all nnUNet's interface arguments
    client = FlowerClient(task=args.task, args=args, device=device)

    fl.client.start_client(
        server_address=f"0.0.0.0:{args.port}",
        client=client.to_client(),  # <-- where FlowerClient is of type flwr.client.NumPyClient object
        grpc_max_message_length=2147483647,
    )

    # Clean up after federated training and perform local validation
    if args.task == "train":
        client.trainer.on_train_end()
        client.trainer.perform_actual_validation()


if __name__ == "__main__":

    client = FlowerClient(task=args.task, args=args)

    fl.client.start_client(
        server_address=f"0.0.0.0:{args.port}",
        client=client.to_client(),  # <-- where FlowerClient is of type flwr.client.NumPyClient object
        grpc_max_message_length=2147483647,
    )

    # Clean up after federated training and perform local validation
    if args.task == "train":
        client.trainer.on_train_end()
        client.trainer.perform_actual_validation()
