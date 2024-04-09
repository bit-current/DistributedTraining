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

address_store = ChainMultiAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Create an instance of DHTManager
dht_manager = DHTManager(
    address_store=address_store,
    bittensor_network=BittensorNetwork,
    my_uid=my_uid,
    dht_host_address=args.miner.dht_host_address,
    dht_tcp_port=args.miner.dht_tcp_port,
    dht_udp_port=args.miner.dht_udp_port,
    dht_external_ip=args.miner.dht_external_ip,
    dht_private_key=args.miner.dht_private_key, 
    my_hotkey=my_hotkey,
    store=True
)

# Use the DHTManager to manage DHT interactions
dht_manager.manage_dht()
dht_addresses = test = " --- ".join([str(multi) for multi in dht_manager.my_dht.get_visible_maddrs()])
logger.info(f"DHT address at:{dht_addresses}")

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

def serialize_gradients(aggregated_gradients):
    # Serialize gradients to a byte stream
    buffer = io.BytesIO()
    torch.save(aggregated_gradients, buffer)
    buffer.seek(0)  # Move to the start of the buffer
    return buffer.getvalue()

# Hypothetical function to send gradients
def send_gradients(aggregated_gradients, storage_dht = dht_manager.my_dht, hotkey = my_hotkey):
    # Hypothetical sending function
    logging.info("Uploading gradients to DHT.")
    serialized_gradients = serialize_gradients(aggregated_gradients)
    stored = storage_dht.store(hotkey, serialized_gradients, time.time()+3600)
    if not stored:
        logging.warning(f"DHT storage failed for hotkey: {hotkey}")
    # Implement sending logic here

# Initialize aggregated gradients
aggregated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

# Training loop
last_send_time = time.time()
optimizer.zero_grad()
for epoch in range(epochs):
    for step, batch in enumerate(data_loader):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        
        # Aggregate gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                aggregated_gradients[name] += param.grad
                
        # Zero gradients for the next accumulation
        optimizer.step()
        optimizer.zero_grad()

        # Check if it's time to send the gradients
        current_time = time.time() ##time.time()%sth works better
        if current_time - last_send_time >= send_interval:
            try:
                send_gradients(aggregated_gradients, dht_manager.my_dht, my_hotkey)
            except:
                continue
            last_send_time = current_time
            # Reset aggregated gradients
            aggregated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}