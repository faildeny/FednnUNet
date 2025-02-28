import argparse
import json
import subprocess

# Convenience script to run federated training on a multi-gpu cluster
# Each node (data-center) is spawned on a determined GPU and communicates with server on the provided network port

parser = argparse.ArgumentParser()

parser.add_argument(
    "task",
    type=str,
    help="Determines the task to be performed. Options are: extract_fingerprint, plan_and_preprocess or train",
)

parser.add_argument(
    "data_centers",
    type=lambda a: json.loads("[" + a.replace(" ", ",") + "]"),
    default="",
    help="List of dataset ids" " (data centers) for federated training",
)

parser.add_argument(
    "configuration", type=str, help="Configuration that should be trained"
)
parser.add_argument(
    "fold",
    type=str,
    nargs="?",
    default=None,
    help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.",
)
parser.add_argument(
    "--gpu_memory_target",
    type=lambda a: json.loads("[" + a.replace(" ", ",") + "]"),
    default="",
    help="GPU memory target in GB"
    " for each dataset, must have the same length as data_centers",
)
parser.add_argument(
    "--port", type=int, required=True, help="Port number for the server to listen on"
)

args, unknown = parser.parse_known_args()

datasets = set(args.data_centers)
num_clients = len(datasets)
task = args.task
fold = args.fold

gpu_memory_target = None

if args.gpu_memory_target:
    gpu_memory_target = args.gpu_memory_target
    if len(gpu_memory_target) != num_clients:
        raise ValueError("gpu_memory_target must have the same length as data_centers")
    if task != "plan_and_preprocess":
        print(
            f"WARNING: {task} task does not accept gpu_memory_target argument. It will be ignored."
        )
    # Create a dictionary with the dataset id as key and the gpu memory target as value
    gpu_memory_target_mapping = dict(zip(datasets, gpu_memory_target))

if task == "extract_fingerprint" or task == "plan_and_preprocess":
    folds = [0]
elif fold == "all":
    folds = list(range(5))
elif fold is None:
    raise ValueError("Fold must be specified for the {task} task")
else:
    folds = [int(fold)]

configuration = args.configuration
port = args.port

multi_gpu = True

# mnms dataset
# node_mapping = {301: 2, 302: 2, 303: 3, 304: 3, 305: 4}
# fetal dataset
node_mapping = {1: 0, 2: 1, 3: 1, 5: 0}
process_prefix = ""

for fold in folds:
    print(f"Starting {task} for fold {fold}")
    try:
        print("Starting server")
        if multi_gpu:
            process_prefix = "CUDA_VISIBLE_DEVICES=0"
        server_process = subprocess.Popen(
            f"{process_prefix} python fednnunet/server.py {task} -n {num_clients} --port {port}",
            shell=True,
            stderr=subprocess.PIPE,
            text=True,
        )
        # server_process = subprocess.Popen(f"python server.py {config_path}", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # print(server_process.stdout)
        # Break when "ready" is printed
        for line in server_process.stderr:
            print(line, end="")  # process line here
            if "Requesting initial parameters" in line:
                break

        client_processes = []
        for client_dataset in datasets:
            print("Starting client " + str(client_dataset))
            if multi_gpu:
                gpu = node_mapping[client_dataset]
                process_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
                print(
                    f"Running {task} for dataset {client_dataset} with fold {fold} on GPU {gpu}"
                )

            optional_args = ""
            # pass the undefined arguments to the client
            if unknown:
                optional_args += " ".join(unknown) + " "
            if gpu_memory_target:
                optional_args += (
                    f"-gpu_memory_target {gpu_memory_target_mapping[client_dataset]} "
                )

            if task == "plan_and_preprocess":
                command = f"{process_prefix} python fednnunet/client_entrypoints.py --port {port} {task} -d {client_dataset} {optional_args}"
            elif task == "train":
                command = f"{process_prefix} python fednnunet/client_entrypoints.py --port {port} {task} {client_dataset} {configuration} {fold} {optional_args}"
            print(command)
            client_processes.append(subprocess.Popen(command, shell=True))

        for line in server_process.stderr:
            print(line, end="")

        server_process.wait()

    except KeyboardInterrupt:
        server_process.terminate()
        server_process.wait()
        for client_process in client_processes:
            client_process.terminate()
            client_process.wait()

        print("Server and clients stopped")
