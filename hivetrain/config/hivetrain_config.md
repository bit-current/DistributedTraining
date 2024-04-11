 # Bittensor Meta-Miner and Torch Miner Arguments Documentation

## Table of Contents
1. [Introduction](#intro)
2. [Importing Necessary Libraries and Modules](#imports)
3. [Defining Argument Parsers for Meta-Miner and Torch Miner](#args)
4. [`add_meta_miner_args` Function](#add-meta-miner-args)
5. [`add_torch_miner_args` Function](#add-torch-miner-args)
6. [`add_orchestrator_args` Function](#add-orchestrator-args)
7. [Conclusion](#conclusion)

<a name="intro"></a>
## Introduction
This documentation explains the logic behind the provided Python code, which consists of several functions and imports necessary for running Bittensor Meta-Miner and Torch Miner applications. The code snippet is structured to add custom arguments to `argparse` for easier configuration of these applications.

<a name="imports"></a>
## Importing Necessary Libraries and Modules
First, the script imports essential libraries and modules:

1. `os`: Operating system dependent functionality
2. `torch`: PyTorch machine learning library
3. `argparse`: For parsing command-line arguments
4. `bittensor as bt`: Bittensor framework for decentralized machine learning
5. `logger from loguru`: For logging messages

<a name="args"></a>
## Defining Argument Parsers for Meta-Miner and Torch Miner
The script then defines three functions:

1. `add_meta_miner_args(parser)`
2. `add_torch_miner_args(parser)`
3. `add_orchestrator_args(parser)`

These functions are used to add specific arguments for Meta-Miner, Torch Miner, and Orchestrator respectively.

<a name="add-meta-miner-args"></a>
## `add_meta_miner_args` Function
The `add_meta_miner_args(parser)` function sets up the arguments for Meta-Miner:

1. **Boolean flag -** `--meta-miner.log-activity`: Displays logging message every request
2. **String argument -** `--meta-miner.orchestrator-url`: URL of the orchestrator
3. **String default argument -** `--miner-script`: The miner script to execute for training (default value: "miner_cpu.py")
4. **Integer arguments -** `--miner.batch-size`, `--miner.epochs`: Batch size per forward/backward pass and number of epochs to train
5. **String list arguments -** `--miner.validator-urls`, `--miner.tcp-store-address`: URLs of the validators for local testing only (accepts multiple values)
6. **String argument -** `--bootstrapping_server`: Bootstrapping server address
7. **String argument -** `--flask.host_address`: URLs of the validators for local testing only
8. **Integer argument -** `--flask.host_port`: URLs of the validators for local testing only

<a name="add-torch-miner-args"></a>
## `add_torch_miner_args(parser)` Function
The `add_torch_miner_args(parser)` function sets up the arguments for Torch Miner:

1. **Integer argument -** `--rank`: Rank of process/node in training run
2. **Integer argument -** `--world-size`: Number of processes/nodes in training run
3. **String argument -** `--store-address`: IP/URL of the TCPStore
4. **Integer argument -** `--store-port`: Port of the test TCPStore
5. **List argument -** `--initial_peers`: Add a peer. Can be used multiple times to pass multiple peers.
6. **Integer argument -** `--batch_size`: The largest batch size able to fit on your GPU.
7. **Integer argument -** `--save_every`: Save the model every X global steps.

<a name="add-orchestrator-args"></a>
## `add_orchestrator_args(parser)` Function
The `add_orchestrator_args(parser)` function sets up the arguments for Orchestrator:

1. **Integer argument -** `--port`: Port for the orchestrator
2. **String argument -** `--host-address`: Host address for the orchestrator

<a name="conclusion"></a>
## Conclusion
These functions provide a clear way to define and parse custom arguments specific to Bittensor Meta-Miner, Torch Miner, and Orchestrator. This modular approach makes it easy to configure these applications based on the required settings.