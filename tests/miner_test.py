import random
from functools import partial
from ipaddress import ip_address

import hivemind
import requests
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)


# Define encoding function
def encode(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')


# Create dataset and model, same as in the basic tutorial
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = "cpu"

opt = torch.optim.AdamW(model.parameters(), lr=0.001)

request = requests.get("https://api.ipify.org")
request.raise_for_status()

address = request.text
print(f"Received public IP address of this machine: {address}")
version = ip_address(address).version
announce_maddrs = [f"/ip{version}/{address}/tcp/{8009}"]
print(announce_maddrs)
# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(
    host_maddrs=[f"/ip4/0.0.0.0/tcp/{8009}"],
    announce_maddrs = announce_maddrs,
    start=True
)
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=20,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=3000,    # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,            # wrap the SGD optimizer defined above
    scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=15.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=60.0,   # give up on averaging if not successful in this many seconds
    verbose=True,              # print logs incessently
)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", 'wikitext-2-v1', split='train')
encoded_dataset = dataset.map(encode, batched=True)

chunk_size = 300
num_iterations = len(dataset) // chunk_size

opt.load_state_from_peers()

#Get the total number of samples in the dataset
num_samples = len(encoded_dataset)

for i in range(num_iterations):
    print("Iteration:", i, "with new dataset indices")
    # Select random dataset indices for the current chunk
    dataset_indices = random.sample(range(num_samples), chunk_size)

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

        #average_loss = total_loss / len(dataloader)
        current_loss = loss / (step + 1)
        print(f"Current Loss: {current_loss}")
        #print(f"Final Average Loss: {average_loss}")