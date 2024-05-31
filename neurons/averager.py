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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import Repository, HfFolder
from hivetrain.averaging_logic import ParameterizedAverager
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.training_manager import FeedforwardNN
from hivetrain.hf_manager import HFManager

# Assuming `model` is your PyTorch model, `scores` contain your keys and their respective scores,
# and `get_weights(key)` is a function to retrieve serialized gradients
from torchvision import transforms, datasets
from datasets import load_dataset
from bittensor import logging

logging.enable_debug()

args = Configurator.combine_configs()
BittensorNetwork.initialize(args, ignore_regs=True)

my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
#my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = ChainMultiAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Define your model's local directory and repository ID
#local_dir = "./save_me"#args.averager.save_directory #TODO add me to config :)
#repo_id = "test_me"#args.averager.hf_repository #TODO add me to config :)

batch_size = args.batch_size
epochs = 30_000_000_000_000_000
learning_rate = 5e-5
send_interval = 120   # Every 60 seconds 

# Load model and tokenizer
# Load the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Assuming you want to use the 'train' split of the dataset
texts = dataset['test']['text'][:100]

# Load model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.train()

class WikitextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = input_ids.clone()  # Copy input_ids to labels
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Create the dataset and data loader
wikitext_dataset = WikitextDataset(texts, tokenizer)
test_loader = DataLoader(wikitext_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_manager = HFManager(my_repo_id = None, device=device, averaged_model_repo_id= args.storage.averaged_model_repo_id)

averager = ParameterizedAverager(model=model,device=device,hf_manager=hf_manager, local_dir=args.storage.model_dir, gradients_dir=args.storage.gradient_dir ,chain_manager=address_store,bittensor_network=BittensorNetwork, hf_token=os.environ.get("HF_TOKEN"))

averager.run_periodic_averaging(test_loader, 7,0.01,1200)
