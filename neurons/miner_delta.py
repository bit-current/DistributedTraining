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

from huggingface_hub import Repository, HfFolder

import numpy as np
import requests
import torch
from datasets import load_dataset
from hivemind.utils.networking import log_visible_maddrs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hivemind import DHT
from hivetrain.btt_connector import (
    BittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
)
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain.dht_connector import DHTManager

from hivetrain import __spec_version__
#from loguru import logger
from bittensor.btlogging import logging

import time
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

logger = logging
#logger = logging.getLogger()
logger.info("Starting !")


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

BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

# Parameters
model_name = "mekaneeky/tiny-random-gpt2"
batch_size = 30
epochs = 30_000_000_000_000_000
learning_rate = 5e-5
send_interval = 60   # Every 60 seconds 

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
model.train()

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        return {key: val.squeeze() for key, val in encoding.items()}  # Remove batch dimension

# Custom collate function (might be optional if __getitem__ returns the correct format)
def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = input_ids.clone()  # Adjust this based on your specific needs
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Dataset & DataLoader
dataset_texts = ["Sample text 1", "Sample text 2", "Sample text 3"]  # example dataset
dataset = MyDataset(dataset_texts, tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)


def upload_weight_deltas(model, base_weights_path="base_weights.pt", trained_weights_path="trained_weights.pt", hf_repo_path="hf_repo_name", hf_repo_url="your_hf_repo_url"):
    # Ensure authentication token is available
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login`.")

    # Calculate weight deltas
    base_weights = torch.load(base_weights_path)
    torch.save(model.state_dict(), trained_weights_path)
    trained_weights = torch.load(trained_weights_path)
    weight_deltas = {name: trained_weights[name] - base_weights[name] for name in trained_weights}

    # Save weight deltas locally
    deltas_path = "weight_deltas.pt"
    torch.save(weight_deltas, deltas_path)

    # Initialize repository (assumes repository already exists)
    repo = Repository(local_dir=hf_repo_path, clone_from=hf_repo_url, use_auth_token=True)
    
    # Commit and push weight deltas to Hugging Face
    repo.git_pull()  # Ensure the local repo is up to date
    repo.add(deltas_path)
    repo.commit("Update weight deltas")
    repo.push()

# Initialize aggregated gradients
aggregated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}


def load_and_compare_weights(model, base_weights_path="base_weights.pt", trained_weights_path="trained_weights.pt", hf_repo_path="your_hf_repo_path", use_auth_token=True):
    # Load base weights
    base_weights = torch.load(base_weights_path)

    # Save current model weights as the "trained" weights
    torch.save(model.state_dict(), trained_weights_path)

    # Calculate delta between base and trained weights
    trained_weights = torch.load(trained_weights_path)
    weight_deltas = {name: trained_weights[name] - base_weights[name] for name in trained_weights}

    # Save weight deltas
    delta_path = f"{hf_repo_path}/weight_deltas.pt"
    torch.save(weight_deltas, delta_path)

    # Initialize and update the HF repository
    repo = Repository(local_dir=hf_repo_path, use_auth_token=use_auth_token)
    repo.git_add("weight_deltas.pt")
    repo.git_commit("Update weight deltas")
    repo.git_push()

# Example of integrating the delta calculation and storage into the training loop
last_send_time = time.time()
send_interval = 3600  # Define your send interval here
optimizer.zero_grad()
for epoch in range(epochs):
    for step, batch in enumerate(data_loader):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_time = time.time()
        if current_time - last_send_time >= send_interval:
            # Instead of sending gradients, calculate and store the weight deltas
            load_and_compare_weights(model, base_weights_path="base_weights.pt", trained_weights_path="trained_weights.pt", hf_repo_path="your_hf_repo_path", use_auth_token=True)
            last_send_time = current_time
            # Optionally, update the base weights to the current weights after sending the deltas
            torch.save(model.state_dict(), "base_weights.pt")