 # Bittensor Network Module Documentation

This documentation describes the logic and functionality of the `BittensorNetwork` Python module. The module is designed to interact with the Bittensor network, managing wallets, subtensors, metagraphs, configurations, and various network-related tasks.

## Contents
1. [Import Statements](#import-statements)
2. [Initialization](#initialization)
3. [Functions](#functions)
   * [initialize_bittensor_objects()](#initialize_bittensor_objects)
   * [check_registered(netuid)](#check_registered)
   * [resync_metagraph()]
   * [should_sync_metagraph(last_sync_time, sync_interval)]
   * [sync(last_sync_time, sync_interval, config)]
   * [serve_extrinsic()]
   * [serve_axon(netuid, host_address, external_address, host_port, external_port)]
4. [BittensorNetwork Class](#bittensornetwork-class)
   * [__new__(cls)]
   * [initialize(config)]
   * [set_weights(scores)]
   * [should_set_weights()]
   * [detect_metric_anomaly(metric="loss", OUTLIER_THRESHOLD=2, MEDIAN_ABSOLUTE_DEVIATION=True)]
   * [run_evaluation()]
   * [rate_limiter(public_address, n=10, t=60)]

## <a name="import-statements"></a>Import Statements

The module imports the following libraries and modules:

```python
import bittensor as bt
import copy
import math
import numpy as np
import torch
import time
from typing import List, Tuple
import bittensor.utils.networking as net
import threading
import logging
from . import __spec_version__
```

## <a name="initialization"></a>Initialization

The `BittensorNetwork` module initializes the necessary objects for interacting with the Bittensor network. These objects include wallets, subtensors, metagraphs, and configurations. The initialization process checks if a hotkey is registered on the specified netuid, creates new objects if not, and sets up various locks for thread safety.

## <a name="functions"></a>Functions

### `initialize_bittensor_objects()`

This function initializes the Bittensor wallet, subtensor, metagraph, and configuration based on the provided base config. If the config's mock flag is set to True, mock objects are created instead of real ones for testing purposes.

### `check_registered(netuid)`

Checks if a hotkey with the specified netuid is registered on the current Subtensor network and prints an error message if not.

### `resync_metagraph()`

Resynchronizes the metagraph with the latest state from the Bittensor network, updating the local copy of the metagraph object.

### `should_sync_metagraph(last_sync_time, sync_interval)`

Determines if the metagraph should be synced based on the last sync time and specified sync interval.

### `sync(last_sync_time, sync_interval, config)`

Synchronizes the metagraph with the latest state from the Bittensor network if it's been enough time since the last sync. The function also handles any errors or exceptions that may occur during synchronization.

### `serve_extrinsic()`

Subscribes a Bittensor endpoint to the subtensor chain by serving an extrinsic with the specified parameters. The function checks if the axon information is up-to-date before attempting to serve and returns True if successful or False otherwise.

### `serve_axon(netuid, host_address, external_address, host_port, external_port)`

Initializes a new Axon instance with the specified parameters and serves it on the network using the Subtensor API. The function returns the newly created Axon object.

## <a name="bittensornetwork-class"></a>BittensorNetwork Class

The `BittensorNetwork` class is a singleton designed to manage various aspects of the Bittensor network, such as wallets, subtensors, metagraphs, configurations, and thread safety.

### `__new__(cls)`

Initializes a new instance of the `BittensorNetwork` class, creating instances of the required objects (wallet, subtensor, metagraph, and config), registering the hotkey on the network if necessary, and setting up locks for thread safety.

### `initialize(config)`

Initializes the Bittensor wallet, subtensor, metagraph, and configuration based on the provided base config, just like the `initialize_bittensor_objects()` function.

### `set_weights(scores)`

Sets the neuron weights with the specified scores on the Subtensor network. The function processes the raw scores to fit within Subtensor's limitations and sends them to the network for storage.

### `should_set_weights()`

Determines if the neuron weights should be updated based on the current block and last update of the metagraph. Returns a boolean value indicating whether it's time to set new weights.

### `detect_metric_anomaly(metric="loss", OUTLIER_THRESHOLD=2, MEDIAN_ABSOLUTE_DEVIATION=True)`

Detects anomalies in the specified metric for each miner by calculating their scores based on whether they are outliers. Returns a dictionary with public addresses as keys and their corresponding scores as values.

### `run_evaluation()`

Evaluates the miners based on their model checksum consensus or metric anomalies, setting new weights accordingly if necessary. Clears both the `model_checksums` and `metrics_data` dictionaries after each evaluation.

### `rate_limiter(public_address, n=10, t=60)`

Checks if the specified public address has exceeded the maximum number of requests within the given time window. If so, it adds the address to a blacklist for the specified time period. Otherwise, it allows the request to proceed.