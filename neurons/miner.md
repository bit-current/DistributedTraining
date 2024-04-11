 # HiveMiner Python Code Documentation

This Python script is designed for training a machine learning model using PyTorch and Bittensor's Hivemind decentralized training platform. The code imports necessary libraries, sets up configurations, and trains the model in a distributed manner.

## Importing Libraries
```python
import argparse                     # For command-line arguments
import ipaddress                   # For IP address manipulation
import logging                      # For logging
import os                          # For file system operations
import random                       # For random number generation
import re                          # For regular expressions
import sys                         # For interacting with the Python interpreter
from functools import partial        # For creating function wrappers
import math                        # For mathematical functions
import bittensor as bt             # Bittensor library for Hivemind integration
from bittensor import metagraph     # Import Metagraph module for handling network connections

import numpy as np                # For numerical operations
import requests                    # For making HTTP requests
import torch                       # PyTorch library for machine learning and deep learning
from datasets import load_dataset    # For loading remote datasets
from hivemind.utils.networking import log_visible_maddrs
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch import LightningModule, LightningTrainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hivetrain.btt_connector import (
    BittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
)
from hivetrain.config import Configurator
```

## Configuration
```python
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
logger = logging.getLogger("lightning.pytorch")

args = Configurator.combine_configs()

# ... (rest of the code)
```

This section initializes the logging system and merges various configuration files into a single `args` object, which will be used throughout the script.

## Helper Functions
```python
def flatten_list(nested_list):
    """Flattens a nested list."""
    if nested_list and isinstance(nested_list[0], list):
        # Assumes only one level of nesting
        return [item for sublist in nested_list for item in sublist]
    return nested_list

# ... (rest of the code)
```

This section defines a helper function `flatten_list()`, which is used to flatten lists that may have one level of nesting.

## Basic Configuration Values
```python
inital_peers_request = requests.get(args.miner.bootstrapping_server)
initial_peers = inital_peers_request.json()["initial_peers"]
assert not (initial_peers is None)
# initial_peers = flatten_list(args.initial_peers)
batch_size = args.batch_size
save_every = args.save_every
block_size = 512
num_steps = 100_000_000_000 # infinite training
target_batch_size = 81920 # when to average all weights.

dataset_config = {
    "dataset": "tiiuae/falcon-refinedweb",
    "key": "content",
    "split": "train",
    "block_size": block_size,
}
```

This section sets some basic configuration values such as batch size, save interval, initial peers list from the `args` object, and dataset-related configurations.

## Initializing Model Components
```python
config = AutoConfig.from_pretrained(
    "gpt2",
    n_embd=block_size,
    n_ctx=block_size,
    n_layer=2,
    n_head=2,
    n_positions=block_size,
    n_inner=block_size * 4,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    summary_first_dropout=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.05,
    summary_type="cls_index",
    summary_proj_to_labels=True,
    summary_use_proj=True,
    torch_dtype=torch.bfloat16,
)

print(config)

model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2",
    cache_dir="/tmp/tokenizer",
    padding="max_length",
    padding_side="left",
    use_fast=True,
    return_overflowing_tokens=True,
    truncation=True,
)
tokenizer.pad_token = tokenizer.eos_token
```

This section initializes and loads the model components using Hugging Face's `AutoConfig` and `AutoModelForCausalLM` classes. Additionally, it loads the tokenizer for converting text to tensors.

## Dataset Handling
```python
class StreamingDataModule(LightningDataModule):
    # ... (class definition)

class StreamingDataset(IterableDataset):
    # ... (class definition)

dataset = StreamingDataModule(tokenizer, dataset_config)
```

This section defines two custom classes `StreamingDataModule` and `StreamingDataset` for handling streaming datasets. The `StreamingDataModule` class wraps the `StreamingDataset` instance to be compatible with PyTorch's `LightningDataModule`.

## Model Training
```python
class MinerTrainer(LightningModule):
    # ... (class definition)

hparams = dict(
    learning_rate=0.001,
    weight_decay=0.1,
    eps=1e-8,
    warmup_steps=10,
    batch_size=batch_size,
    num_steps=num_steps,
    block_size=block_size,
)

strategy = HivemindStrategy(
    run_id=f"hiveminer_{str(__spec_version__)}",
    batch_size=batch_size,
    target_batch_size=target_batch_size,
    initial_peers=initial_peers,
    use_ipfs=False,
    use_relay=True,
    use_auto_relay=True,
    verbose=False,
    wait_timeout=180,
    bootstrap_timeout=135,
    matchmaking_time=360.0,
    averaging_timeout=600.0,
    delay_state_averaging=True,
    delay_grad_averaging=True,
    delay_optimizer_step=True,
    offload_optimizer=True,
    reuse_grad_buffers=False,
    # grad_compression=Float16Compression(),
    # state_averaging_compression=Float16Compression(),
    # load_state_compression=NoCompression(),
    # scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),
)

visible_addresses = [
    str(a)
    for a in strategy.dht.get_visible_maddrs()
    if not ipaddress.ip_address(a.values()[0]).is_loopback
]

log_visible_maddrs(strategy.dht.get_visible_maddrs(), only_p2p=False)
# my_ids = []
# pattern = r"(/p2p/.*)"
# for peer in list(visible_addresses):
#     match = re.search(pattern, peer)
#     if match:
#         my_ids.append(match.group(1))

# for peer in list(set(my_ids)):
#     print(f"PEER-ID: {peer}")

params = set_trainable_parameters(model, hparams)
optimizer = AdamW(params, lr=hparams["learning_rate"], eps=hparams["eps"])

class MinerConsoleLogging(Callback):
    # ... (class definition)

class MinerModelSaver(Callback):
    # ... (class definition)

class ValidationCommunicator(Callback):
    # ... (class definition)

train_params = dict(
    accelerator="auto",
    strategy=strategy,
    devices="auto",
    max_steps=num_steps * target_batch_size,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=1,
    precision="32-true",
    accumulate_grad_batches=1,  # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    enable_progress_bar=False,
    callbacks=[],
)

# Set weights as trainable (looks like this is useless)
# def set_trainable_parameters(model, hparams):
#     no_decay = ["bias", "LayerNorm.weight"]
#     grouped_parameters = []

#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         if any(nd in n for nd in no_decay):
#             weight_decay = 0.0
#         else:
#             weight_decay = hparams["weight_decay"]

#         grouped_parameters.append(
#             {
#                 "params": [p],
#                 "weight_decay": weight_decay,
#             }
#         )

# Set model parameters as trainable
params = set_trainable_parameters(model, hparams)

optimizer = AdamW(params, lr=hparams.get("learning_rate", 0.001), eps=hparams.get("eps", 1e-8))

train_model = MinerTrainer(model, optimizer, hparams)

trainer = Trainer(**train_params)
trainer.fit(train_model, dataset)
```

This section initializes the Hivemind strategy and sets up various training-related configurations. Then, it creates a PyTorch trainer instance and trains the model using the provided dataset with the defined `MinerTrainer` class.