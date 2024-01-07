
import torch
import os
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

# Define encoding function
def encode(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')


# Create dataset and model, same as in the basic tutorial
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = "cpu"

opt = torch.optim.AdamW(model.parameters(), lr=0.00001)

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(initial_peers=["/ip4/54.89.124.220/tcp/8008/p2p/12D3KooWRBaYDFiGsTuReihRtn9voVV1wAMxKzeeU9WDPSZjk9aX"], 
                   start=True)

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=10,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=200,    # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,            # wrap the SGD optimizer defined above
    scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=15.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=60.0,   # give up on averaging if not successful in this many seconds
    verbose=True              # print logs incessently
)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", 'wikitext-2-v1', split='train')
encoded_dataset = dataset.map(encode, batched=True)

for i in range(30):
    print("Iteration:", i, "with new dataset indices")
    # Select random dataset_indices to use for training of set size 200
    dataset_indices = [i for i in range(0, len(dataset))]
    # Choose randomly 200 indices
    dataset_indices = torch.randperm(len(dataset_indices))[:200].tolist()
    # Select dataset indices to use for optimization step
    sub_dataset = encoded_dataset.select(dataset_indices)

    # Create a PyTorch DataLoader
    dataloader = DataLoader(sub_dataset, batch_size=10, collate_fn = default_data_collator)

    # Train data for one epoch
    for step, batch in enumerate(dataloader):
        
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
        
        loss.backward()
        # Adjust gradient
        opt.step()

        print(opt.state_averager.is_alive())
        #print(opt.tracker.is_alive())
        print(opt.grad_averager)
        #print("weights post:", model.weight.data[...])
    time.sleep(10)