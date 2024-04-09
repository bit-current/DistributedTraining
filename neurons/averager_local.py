import torch
import io
import argparse
import ipaddress
import os
import random
import re
import sys
import time
from functools import partial
from math import isnan
import bittensor as bt
from bittensor import metagraph

import numpy as np
import requests
import torch
from datasets import load_dataset
from hivemind.utils.networking import log_visible_maddrs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import Repository, HfFolder
from hivetrain.averaging_logic import LocalAverager
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
from hivetrain.chain_manager import LocalAddressStore
from hivetrain.dht_connector import DHTManager
# Assuming `model` is your PyTorch model, `scores` contain your keys and their respective scores,
# and `get_weights(key)` is a function to retrieve serialized gradients

args = Configurator.combine_configs()

BittensorNetwork.initialize(args)

my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = LocalAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Create an instance of DHTManager
dht_manager = DHTManager(
    address_store=address_store,
    bittensor_network=BittensorNetwork,
    my_uid=my_uid,
    my_hotkey = my_hotkey,
    dht_host_address=args.miner.dht_host_address,
    dht_tcp_port=args.miner.dht_tcp_port,
    dht_udp_port=args.miner.dht_udp_port,
    dht_external_ip=args.miner.dht_external_ip,
    dht_private_key=args.miner.dht_private_key,
    store=False
)

# Use the DHTManager to manage DHT interactions
dht_manager.manage_dht()

# Define your model's local directory and repository ID
local_dir = "./save_me"#args.averager.save_directory #TODO add me to config :)
repo_id = "test_me"#args.averager.hf_repository #TODO add me to config :)
model_name = "mekaneeky/tiny-random-gpt2"

# Save the updated model
# model.save_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

averager = Averager(model=model, local_dir="./save_me", repo_id=repo_id, dht=dht_manager.my_dht,bittensor_network=BittensorNetwork, hf_token=os.environ.get("HF_TOKEN"))
averager.run_periodic_averaging(60)
# # Push the model to the Hugging Face Hub
# push_to_hf_hub(local_dir=local_dir, repo_id=repo_id, hf_token=args.averager.hf_token, commit_message=f"Updated model SN25 with {222}")#FIXME add block numbers
