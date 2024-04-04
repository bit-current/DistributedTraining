import torch
import io
import math
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
from hivetrain import __spec_version__
#from loguru import logger
from bittensor.btlogging import logging
from hivetrain.validation_logiv import ModelValidator

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
    dht_private_key=args.miner.dht_private_key
)

# Use the DHTManager to manage DHT interactions
dht_manager.manage_dht()

# Hypothetical function to send gradients
def receive_gradients(aggregated_gradients, storage_dht = dht_manager.my_dht, hotkey = my_hotkey):
    # Hypothetical sending function
    serialized_gradients = storage_dht.get(hotkey)
    aggregated_gradients = deserialize_gradients(serialized_gradients)
    
    return aggregated_gradients
    # Implement sending logic here

def deserialize_gradients(serialized_gradients):
    buffer = io.BytesIO(serialized_gradients)
    buffer.seek(0)
    return torch.load(buffer)

def update_model_weights(model, gradients):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in gradients:
                param -= gradients[name]  # Update the model with the gradient

def test_model(model, data_loader, criterion, metric='loss'):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            labels = batch['input_ids']
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Multiply by batch size
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    if metric == 'perplexity':
        return math.exp(average_loss)
    else:
        return average_loss

# Load your model and other necessary components here

original_state_dict = model.state_dict()  # Save the original state dict
base_loss = test_model(model, test_data_loader, criterion, metric='loss')
base_perplexity = test_model(model, test_data_loader, criterion, metric='perplexity')
print(f"Base model loss: {base_loss}")
print(f"Base model perplexity: {base_perplexity}")

validator = ModelValidator(model, data_loader, criterion, interval=3600)
validator.start_periodic_validation()


# scores = {}
# for key in public_keys:
#     if existence_check(key):
#         serialized_gradients = get_weights(key)
#         gradients = deserialize_gradients(serialized_gradients)
#         update_model_weights(model, gradients)
#         loss_after_update = test_model(model, test_data_loader, criterion, metric='loss')
#         perplexity_after_update = test_model(model, test_data_loader, criterion, metric='perplexity')
#         loss_score = max(0, base_loss - loss_after_update)  # Score based on loss reduction
#         perplexity_score = max(0, base_perplexity - perplexity_after_update)  # Score based on perplexity reduction
#         scores[key] = {'loss_score': loss_score, 'perplexity_score': perplexity_score}
#         print(f"Scores for {key}: Loss Score = {loss_score}, Perplexity Score = {perplexity_score}")
#         model.load_state_dict(original_state_dict)  # Reset to original state dict
