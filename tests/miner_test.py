
import torch
import os
from ipaddress import ip_address
import requests
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import hivemind
from functools import partial
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from tqdm import tqdm

# Define encoding function
def encode(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')


# Create dataset and model, same as in the basic tutorial
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = "cpu"

opt = torch.optim.AdamW(model.parameters(), lr=0.00001)

version = "4"
address = "34.229.118.117"
announce_maddrs = [f"/ip{version}/{address}/tcp/{8009}"]

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(
    host_maddrs=[f"/ip4/0.0.0.0/tcp/{8009}"],
    initial_peers=["/ip4/172.31.38.115/tcp/8008/p2p/12D3KooWJUccTxmAumNy983rahKrm7fmCguAm3ySbHpzSe1sJsCV"], 
    announce_maddrs = announce_maddrs,
    start=True
)

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=10,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=4000,    # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,            # wrap the SGD optimizer defined above
    scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=15.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=60.0,   # give up on averaging if not successful in this many seconds
    verbose=False,              # print logs incessently
)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", 'wikitext-2-v1', split='train')
encoded_dataset = dataset.map(encode, batched=True)

chunk_size = 200
num_iterations = len(dataset) // chunk_size

opt.load_state_from_peers()

for i in range(num_iterations):
    print("Iteration:", i, "with new dataset indices")
    # Select dataset indices for the current chunk
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size
    dataset_indices = [j for j in range(start_index, end_index)]

    # Create a PyTorch DataLoader
    dataloader = DataLoader(encoded_dataset.select(dataset_indices), batch_size=10, collate_fn=default_data_collator)

    pb_bar = tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, leave=False)
    
    total_loss = 0 

    # Train data for one epoch
    for step, batch in pb_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        opt.zero_grad()

        # Forward pass
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            labels = labels
        )     
        # Backward pass    
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        # Adjust gradient
        opt.step()
        
        #print(f"Step {step} Loss: {loss}")
        pb_bar.set_description("Train loop: loss {:.4f} ".format(loss.item()))

    average_loss = total_loss / len(dataloader)
    print(f"Final Average Loss: {average_loss}")

    time.sleep(10)