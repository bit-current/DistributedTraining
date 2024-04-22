import random

import torch

from hivemind import DHT
from hivetrain.btt_connector import (
    LocalBittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
)
from hivetrain.chain_manager import LocalAddressStore
from hivetrain.config import Configurator
from hivetrain.hf_manager import LocalHFManager
from hivetrain.training_manager import MNISTDeltaTrain

from hivetrain import __spec_version__
from bittensor.btlogging import logging

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

logging.info("Starting !")


def flatten_list(nested_list):
    """Flatten a nested list."""
    if nested_list and isinstance(nested_list[0], list):
        # Assumes only one level of nesting
        return [item for sublist in nested_list for item in sublist]
    return nested_list


# # set some basic configuration values
# inital_peers_request = requests.get(args.miner.bootstrapping_server)
# initial_peers = inital_peers_request.json()["initial_peers"]
# assert not (initial_peers is None)
args = Configurator.combine_configs()

LocalBittensorNetwork.initialize(args)
my_hotkey = args.wallet.hotkey#LocalBittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

#address_store = LocalAddressStore(LocalBittensorNetwork.subtensor, args.netuid,LocalBittensorNetwork.wallet)
#address_store.store_hf_repo(args.storage.gradient_dir)

# Parameters

batch_size = args.batch_size
epochs = 10  # Adjust epochs for MNIST training, 30_000_000_000_000_000 is unrealistic
learning_rate = 5e-5
send_interval = 90  # Every 60 seconds

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Training loop
hf_manager = LocalHFManager(repo_id=args.storage.model_dir)
#def __init__(self, model_name, data_loader,gradients_dir, learning_rate=5e-5, send_interval=30):

training_loop = MNISTDeltaTrain(None, train_loader, args.storage.gradient_dir,test_loader=test_loader, send_interval=args.miner.send_interval)
training_loop.save_model()