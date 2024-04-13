import torch
import io
import os
import re
import time
import bittensor as bt

from torch.utils.data import DataLoader, Dataset, IterableDataset

from hivetrain.btt_connector import (
    BittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
)

from torch.optim import AdamW, SGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hivetrain.chain_manager import LocalAddressStore
from hivetrain.config import Configurator
from hivetrain import __spec_version__
from bittensor.btlogging import logging
from hivetrain.validation_logic import MNISTDeltaValidator
from hivetrain.dht_connector import DHTManager
from hivetrain.hf_manager import LocalHFManager
from hivetrain.training_manager import FeedforwardNN

args = Configurator.combine_configs()

BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = LocalAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)


batch_size = args.batch_size
epochs = 10  # Adjust epochs for MNIST training, 30_000_000_000_000_000 is unrealistic
learning_rate = 5e-5
receive_interval = 1  # Every 60 seconds

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = FeedforwardNN()
optimizer = SGD(model.parameters(),lr = 0.1)


# Load your model and other necessary components here
#    def __init__(self, model, optimizer, data_loader, bittensor_network=None, chain_manager=None, interval=3600, local_gradient_dir="local_gradients"):
hf_manager = LocalHFManager(repo_id=args.storage.model_dir)
validator = MNISTDeltaValidator(model=model,optimizer=optimizer, data_loader=test_loader,
    bittensor_network=BittensorNetwork ,hf_manager=hf_manager, interval=receive_interval, chain_manager=address_store,local_gradient_dir=args.storage.gradient_dir,
    )
validator.start_periodic_validation()
