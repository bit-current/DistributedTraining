import os
import time
from functools import partial
from ipaddress import ip_address
import random
import hivemind
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from template.data.dataset import SubsetFalconLoader
import subprocess

# Define encoding function
def encode(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )


# Create dataset and model, same as in the basic tutorial
model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-677m")
device = "cuda"
port = 40317
model = model.to(device=device)
# model = model.to(device=device)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)


request = requests.get("https://api.ipify.org")
request.raise_for_status()

address = request.text
print(f"Received public IP address of this machine: {address}")
version = ip_address(address).version
announce_maddrs = [f"/ip{version}/{address}/tcp/{port}"]

version = "4"
# address = "34.229.118.117"
announce_maddrs = [f"/ip{version}/{address}/tcp/{port}"]

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(
    host_maddrs=[f"/ip4/0.0.0.0/tcp/{port}"],
    # initial_peers=["/ip4/161.97.156.125/tcp/41669/p2p/12D3KooWJWz47mnxvii7ErigQErfBeAhrWBhUTMUVruuEcKxNpMT","/ip4/69.30.85.69/tcp/22085/p2p/12D3KooWDzGzZYn1jFurAZCfpEjjKWXQ1mpSoGRTKofK4n4GidFz","/ip4/69.30.85.69/tcp/22085/p2p/12D3KooWQVbv3krYkkuU8NDdWYGC4cEtXtgHd5df3rW22GnZcSpM"],
    announce_maddrs=announce_maddrs,
    start=True,
)

hivemind.utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

with open('visible_maddrs.txt', 'w') as file:
    for addr in dht.get_visible_maddrs():
        # Assuming each addr needs to be converted to string
        file.write(str(addr) + '\n')  # Add a newline character to separate each address

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,  # use a DHT that is connected with other peers
    run_id="use_local",  # unique identifier of this collaborative run
    scheduler=None,
    batch_size_per_step=1,  # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=300,  # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,  # wrap the SGD optimizer defined above
    use_local_updates=True,  # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=15.0,  # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=120.0,  # give up on averaging if not successful in this many seconds
    verbose=True,  # print logs incessently
    grad_compression=hivemind.Float16Compression(),
    state_averaging_compression=hivemind.Float16Compression(),
)

tokenizer = AutoTokenizer.from_pretrained("kmfoda/gpt2-677m")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
# dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
# encoded_dataset = dataset.map(encode, batched=True)

# chunk_size = 200
# num_iterations = len(dataset) // chunk_size

#opt.load_state_from_peers()

# for i in range(num_iterations):
# print("Iteration:", i, "with new dataset indices")
# # Select dataset indices for the current chunk
# start_index = i * chunk_size
# end_index = (i + 1) * chunk_size
# dataset_indices = [j for j in range(start_index, end_index)]

# Create a PyTorch DataLoader
# dataloader = DataLoader(encoded_dataset.select(dataset_indices), batch_size=10, collate_fn=default_data_collator)
# dataloader = DataLoader(
#     encoded_dataset, batch_size=10, collate_fn=default_data_collator
# )
num_rows = 9000000

for _ in range(1000):
    
    random_indices = random.sample(range(num_rows), 50)
    
    
    dataloader = SubsetFalconLoader(
                batch_size=1, sequence_length=1024, rows=random_indices
            )

    total_loss = 0

    time.sleep(10)

    # Train data for one epoch
    #pb_bar = tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, leave=False)

    # Train data for one epoch
    for step, batch in enumerate(dataloader):
        inputs = batch.to(device)
        
        # Zero gradients
        opt.zero_grad()

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)

        # Backward pass    
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        # Adjust gradient
        opt.step()
        
        #print(f"Step {step} Loss: {loss}")
        #pb_bar.set_description("Train loop: loss {:.4f} ".format(loss.item()))

        #average_loss = total_loss / len(dataloader)
        current_loss = loss / (step + 1)
        print(loss.item(), current_loss)
        
    average_loss = total_loss / step
    print(f"Average Loss: {average_loss}")

print("Shutting down..")
opt.shutdown()
dht.shutdown()
exit()