# Replace vanilla nnUNetTrainer with federated extension
import sys
from typing import Optional, Union

import torch.cuda
from batchgenerators.utilities.file_and_folder_operations import join
from torch.backends import cudnn

import fednnunet.fednnUNetTrainer as nnUNetTrainer
from nnUNet.nnunetv2.run.run_training import (
    get_trainer_from_args,
    maybe_load_checkpoint,
)

sys.modules["nnunetv2.training.nnUNetTrainer.nnUNetTrainer"] = nnUNetTrainer


# Slight modification to the original run_training function from nnUNet that allows to return the prepared trainer object


def run_training(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),
    return_trainer=False,
):

    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if val_with_best:
        assert (
            not disable_checkpointing
        ), "--val_best is not compatible with --disable_checkpointing"

    if num_gpus > 1:
        raise NotImplementedError("FednnUNet does not support DDP training yet.")
    else:
        nnunet_trainer = get_trainer_from_args(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            use_compressed_data,
            device=device,
        )

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
            continue_training and only_run_validation
        ), f"Cannot set --c and --val flag at the same time. Dummy."

        maybe_load_checkpoint(
            nnunet_trainer, continue_training, only_run_validation, pretrained_weights
        )

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if return_trainer:
            return nnunet_trainer

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(
                join(nnunet_trainer.output_folder, "checkpoint_best.pth")
            )
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)
