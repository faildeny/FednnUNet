import os
from io import BytesIO
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


def state_dict_to_bytes(state_dict) -> bytes:
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict) -> Parameters:
    tensors = state_dict_to_bytes(state_dict)
    log(INFO, f"State dict to parameters: {len(tensors)} bytes")
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> dict:
    """Converts bytes back to a PyTorch state_dict."""
    bytes_io = BytesIO(bytes_data)
    return torch.load(bytes_io)


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


def average_dicts(dicts):
    if not dicts:
        return {}

    # Initialize a dictionary to keep track of the sum and count for each key
    totals = {}
    counts = {}

    # Iterate through each dictionary
    for d in dicts:
        for key, value in d.items():
            if key in totals:
                totals[key] += value
                counts[key] += 1
            else:
                totals[key] = value
                counts[key] = 1

    # Calculate the average for each key
    averages = {key: totals[key] / counts[key] for key in totals}

    return averages


def weighted_mean(values, weights):
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def max(values, weights):
    values = torch.tensor(values)
    return torch.max(values).item()


def min(values, weights):
    values = torch.tensor(values)
    return torch.min(values).item()


def concatenate(values, weights):
    # Concatenate the nested lists maintaining the structure
    return [item for sublist in values for item in sublist]


def aggregate_fingerprints(parameters: List[Parameters]) -> Parameters:
    # Extract the state_dicts from the parameters
    state_dicts = [parameters_to_state_dict(p) for p in parameters]

    # Define aggregation functions for each key in the fingerprint
    aggregation_dict = {
        "max": max,
        "min": min,
        "mean": weighted_mean,
        "median": weighted_mean,
        "std": weighted_mean,
        "percentile_00_5": weighted_mean,
        "percentile_99_5": weighted_mean,
        "median_relative_size_after_cropping": weighted_mean,
        "shapes_after_crop": concatenate,
        "spacings": concatenate,
    }

    # Infer number of samples on each client by dict 'shapes_after_crop' length
    num_samples = [len(sd["shapes_after_crop"]) for sd in state_dicts]
    print(f"Num samples per client: {num_samples}")

    new_state_dict = {}
    for key in state_dicts[0].keys():
        if isinstance(state_dicts[0][key], dict):
            new_state_dict[key] = {}
            for subkey in state_dicts[0][key].keys():
                if isinstance(state_dicts[0][key][subkey], dict):
                    new_state_dict[key][subkey] = {}
                    for subsubkey in state_dicts[0][key][subkey].keys():
                        new_state_dict[key][subkey][subsubkey] = aggregation_dict[
                            subsubkey
                        ](
                            [sd[key][subkey][subsubkey] for sd in state_dicts],
                            num_samples,
                        )
                else:
                    new_state_dict[key][subkey] = aggregation_dict[subkey](
                        [sd[key][subkey] for sd in state_dicts], num_samples
                    )
        # elif isinstance(state_dicts[0][key], list):
        #     pass
        else:
            new_state_dict[key] = aggregation_dict[key](
                [sd[key] for sd in state_dicts], num_samples
            )

    # Convert the new state_dict back to parameters
    return state_dict_to_parameters(new_state_dict)


class MyStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        task: str,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

        self.task = task

    def find_common_layers(self, state_dicts):
        # Find the common keys in all state_dicts

        common_keys = set(state_dicts[0].keys())

        for sd in state_dicts[1:]:
            common_keys.intersection_update(sd.keys())
            # Print number of keys in this dictionary
            log(INFO, f"Number of keys in this dictionary: {len(sd.keys())}")

        log(INFO, f"Number of common keys: {len(common_keys)}")

        # Verify dimensions
        compatible_keys = []
        for key in common_keys:
            dimensions = [sd[key].shape for sd in state_dicts]
            if all(dim == dimensions[0] for dim in dimensions):
                compatible_keys.append(key)
        log(INFO, f"Number of compatible keys: {len(compatible_keys)}")

        return compatible_keys

    def create_compatible_state_dict(self, state_dicts, compatible_keys):
        new_state_dict = {}
        for key in compatible_keys:
            # Assuming we take the parameters from the first state_dict
            keys = [s[key] for s in state_dicts]
            new_state_dict[key] = torch.mean(torch.stack(keys, dim=0), dim=0)
        return new_state_dict

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            return [(client, fit_ins) for client in clients]

        parms = [
            parameters_to_state_dict(
                client.get_parameters(
                    ins=fit_ins, timeout=None, group_id=None
                ).parameters
            )
            for client in clients
        ]
        for idx, sd in enumerate(parms):
            torch.save(
                sd,
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), str(idx) + ".arch"
                ),
            )

        compatible_keys = self.find_common_layers(parms)

        # here we are doing the merging already...
        new_state_dict = self.create_compatible_state_dict(parms, compatible_keys)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[List[float], Dict[str, fl.common.Scalar]]:
        if failures:
            fl.common.logger.log(2, f"Round {rnd} had {len(failures)} failures.")

        # Filter out None results due to failures
        successful_results = [result for result in results if result is not None]

        # If there are no successful results, return None or a default value
        if not successful_results:
            fl.common.logger.log(2, f"Round {rnd} had {len(failures)} failures.")
            return None  # or some default values

        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            return (
                aggregate_fingerprints(
                    [res[1].parameters for res in successful_results]
                ),
                {},
            )
        # Perform aggregation on successful results
        aggregated_weights = self.aggregate_weights(successful_results)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return aggregated_weights, average_dicts(
            [r[1].metrics for r in successful_results]
        )

    def aggregate_weights(self, results):

        dicts = [parameters_to_state_dict(res[1].parameters) for res in results]

        compatible_keys = self.find_common_layers(dicts)

        # here we are doing the merging already...
        new_state_dict = self.create_compatible_state_dict(dicts, compatible_keys)

        # Implement weight aggregation logic
        return state_dict_to_parameters(new_state_dict)


# Start Flower server with the custom strategy

import argparse

parser = argparse.ArgumentParser(description="Start Flower server")
parser.add_argument(
    "task",
    type=str,
    help="Determines the task to be performed. Options are: extract_fingerprint, plan_and_preprocess or train",
)
parser.add_argument(
    "-n",
    "--num_clients",
    type=int,
    default=2,
    help="Number of clients to wait for before starting the server",
)
parser.add_argument(
    "--port", type=int, required=True, help="Port number for the server to listen on"
)

args = parser.parse_args()
num_clients = args.num_clients

if args.task == "extract_fingerprint" or args.task == "plan_and_preprocess":
    num_rounds = 1
    fraction_evaluate = 1.0
else:
    # nnUNet's default training length
    num_rounds = 1000
    # Skip federated evaluation to speed up training by one less parameters transfer
    fraction_evaluate = 0.0

strategy = MyStrategy(
    args.task,
    min_available_clients=num_clients,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    fraction_evaluate=fraction_evaluate,
)


# Start Flower server
fl.server.start_server(
    server_address=f"0.0.0.0:{args.port}",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    grpc_max_message_length=2147483647,  # Request a maximum message length to support sending weights from larger, more recent ResEnc architectures
)
