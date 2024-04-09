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

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain import __spec_version__
from bittensor.btlogging import logging
from hivetrain.validation_logic import ModelValidator
from hivetrain.dht_connector import DHTManager

args = Configurator.combine_configs()

BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = ChainMultiAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Use the DHTManager to manage DHT interactions

#FIXME you should be getting from HF
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

# Load your model and other necessary components here

validator = ModelValidator(model=model,optimizer=optimizer, data_loader=data_loader,bittensor_network=BittensorNetwork , interval=120, chain_manager=address_store)
validator.start_periodic_validation()
