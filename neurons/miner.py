from bittensor import logging
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.chain_manager import ChainMultiAddressStore
from hivetrain.config import Configurator
from hivetrain.hf_manager import HFManager
from hivetrain.training_manager import DeltaLoop


logging.enable_debug()
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

address_store = ChainMultiAddressStore(
    BittensorNetwork.subtensor, args.netuid, BittensorNetwork.wallet
)

current_address_in_store = address_store.retrieve_hf_repo(my_hotkey)
logging.info(f"Current value in store:{current_address_in_store}")
if current_address_in_store != args.storage.my_repo_id:
    logging.info(f"Storing new value: {args.storage.my_repo_id}")
    address_store.store_hf_repo(args.storage.my_repo_id)

# Parameters

batch_size = args.batch_size
epochs = 30_000_000_000_000_000
learning_rate = 5e-5
send_interval = 600  # Every 60 seconds

# Load the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Assuming you want to use the 'train' split of the dataset
texts = dataset["train"]["text"]

# Load model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.resize_token_embeddings(len(tokenizer))
# model.train()



class WikitextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze()  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = input_ids.clone()  # Copy input_ids to labels
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Create the dataset and data loader
wikitext_dataset = WikitextDataset(texts, tokenizer)
data_loader = DataLoader(
    wikitext_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
)
# Optimizer
#optimizer = AdamW(model.parameters(), lr=learning_rate)


# hf_manager = HFManager(my_repo_id = args.storage.my_repo_id, averaged_model_repo_id= args.storage.averaged_model_repo_id)
device = "cuda" if torch.cuda.is_available() else "cpu"

hf_manager = HFManager(
    device=device,
    my_repo_id=args.storage.my_repo_id,
    averaged_model_repo_id=args.storage.averaged_model_repo_id,
) 
model_name = "openai-community/gpt2"
training_loop = DeltaLoop(
    device,
    model_name,
    data_loader,
    send_interval=800,
    learning_rate=5e-4,
    hf_manager=hf_manager,
)
training_loop.train(epochs=30_000_000_000_000_000)

