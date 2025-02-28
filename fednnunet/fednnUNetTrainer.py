import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer(nnUNetTrainer):
    '''
    Extension to original nnUNetTrainer enabling epoch by epoch training 
    while sharing the weights with the server.
    '''
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        print("Using federated nnUNetTrainer")
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )

    def run_federated_train_round(self):

        self.on_epoch_start()

        self.on_train_epoch_start()
        train_outputs = []
        for batch_id in range(self.num_iterations_per_epoch):
            train_outputs.append(self.train_step(next(self.dataloader_train)))

        self.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):

                # lrs are the same for all workers so we don't need to gather them in case of DDP training
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()
