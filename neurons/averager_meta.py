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
from hivetrain.averaging_logic import LocalParameterizedAverager
from hivetrain.btt_connector import LocalBittensorNetwork
from hivetrain.config import Configurator
from hivetrain.chain_manager import LocalAddressStore
from hivetrain.dht_connector import DHTManager
from hivetrain.training_manager import FeedforwardNN
from hivetrain.hf_manager import LocalHFManager

# Assuming `model` is your PyTorch model, `scores` contain your keys and their respective scores,
# and `get_weights(key)` is a function to retrieve serialized gradients
from torchvision import transforms, datasets
from datasets import load_dataset


args = Configurator.combine_configs()

LocalBittensorNetwork.initialize(args)

my_hotkey = LocalBittensorNetwork.wallet.hotkey.ss58_address
my_uid = LocalBittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = LocalAddressStore(LocalBittensorNetwork.subtensor, args.netuid,LocalBittensorNetwork.wallet)

# Define your model's local directory and repository ID
#local_dir = "./save_me"#args.averager.save_directory #TODO add me to config :)
#repo_id = "test_me"#args.averager.hf_repository #TODO add me to config :)

batch_size = args.batch_size
epochs = 10  # Adjust epochs for MNIST training, 30_000_000_000_000_000 is unrealistic
learning_rate = 5e-5
receive_interval = 30  # Every 60 seconds

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = FeedforwardNN()


#__init__(self, model, local_dir, bittensor_network=None)
#model, device, chain_manager=None,bittensor_network=None, hf_token=hf_token 
hf_manager = LocalHFManager(repo_id=args.storage.model_dir)
averager = LocalParameterizedAverager(model=model,device="cpu",hf_manager=hf_manager, local_dir=args.storage.model_dir, chain_manager=address_store,bittensor_network=LocalBittensorNetwork, hf_token=os.environ.get("HF_TOKEN"))
#averager.run_periodic_averaging(test_loader,20,300)
averager.run_periodic_averaging(test_loader)
# # Push the model to the Hugging Face Hub
# push_to_hf_hub(local_dir=local_dir, repo_id=repo_id, hf_token=args.averager.hf_token, commit_message=f"Updated model SN25 with {222}")#FIXME add block numbers
