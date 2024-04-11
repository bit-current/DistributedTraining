import random

import torch

from hivemind import DHT
from hivetrain.btt_connector import (
    BittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
)
from hivetrain.chain_manager import LocalAddressStore
from hivetrain.config import Configurator
from hivetrain.hf_manager import LocalHFManager
from hivetrain.training_manager import LocalTrainingLoop

from hivetrain import __spec_version__
from bittensor.btlogging import logging

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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

BittensorNetwork.initialize(args)
my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = LocalAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)
address_store.store_hf_repo(args.storage.gradient_dir)

# Parameters
model_name = "mekaneeky/tiny-random-gpt2"
batch_size = args.batch_size
epochs = 30_000_000_000_000_000
learning_rate = 5e-5
send_interval = 60   # Every 60 seconds 

# Load the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Assuming you want to use the 'train' split of the dataset
texts = dataset['train']['text']

# Load model and tokenizer
model_name = "mekaneeky/tiny-random-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.train()

class WikitextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
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
optimizer = AdamW(model.parameters(), lr=learning_rate)

# def serialize_gradients(aggregated_gradients):
#     # Serialize gradients to a byte stream
#     buffer = io.BytesIO()
#     torch.save(aggregated_gradients, buffer)
#     buffer.seek(0)  # Move to the start of the buffer
#     return buffer.getvalue()

# Initialize aggregated gradients
aggregated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

# Training loop
hf_manager = LocalHFManager(repo_id=args.storage.model_dir)
#def __init__(self, model_name, data_loader,gradients_dir, learning_rate=5e-5, send_interval=30):

training_loop = LocalTrainingLoop("mekaneeky/tiny-random-gpt2", data_loader, learning_rate=5e-4,args.storage.gradient_dir)
training_loop.train(epochs=30_000_000_000_000_000, hf_manager=hf_manager)