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
from datasets import load_dataset

from hivetrain.chain_manager import LocalAddressStore
from hivetrain.config import Configurator
from hivetrain import __spec_version__
from bittensor.btlogging import logging
from hivetrain.validation_logic import LocalValidator
from hivetrain.dht_connector import DHTManager
from hivetrain.hf_manager import LocalHFManager


args = Configurator.combine_configs()

BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = LocalAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Use the DHTManager to manage DHT interactions

#FIXME you should be getting from HF
model_name = "mekaneeky/tiny-random-gpt2"
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
model_name = "mekaneeky/tiny-random-gpt2"
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
data_loader = DataLoader(wikitext_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

# Optimizer


# Load your model and other necessary components here
#    def __init__(self, model, optimizer, data_loader, bittensor_network=None, chain_manager=None, interval=3600, local_gradient_dir="local_gradients"):
hf_manager = LocalHFManager(repo_id=args.storage.model_dir)
validator = LocalValidator(model=model,optimizer=optimizer, data_loader=data_loader,bittensor_network=BittensorNetwork ,hf_manager=hf_manager, interval=30, chain_manager=address_store,local_gradient_dir=args.storage.gradient_dir)
validator.start_periodic_validation()
