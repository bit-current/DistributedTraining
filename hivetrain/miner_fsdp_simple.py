import argparse
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import TCPStore
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import DataLoader
import torch.distributed as dist
import time
import json
import requests
import os
from substrateinterface import Keypair
from hivetrain.config import Configurator
from hivetrain.btt_connector import BittensorNetwork
from datetime import timedelta

# Assuming the Keypair for the miner is generated or loaded here

# def sign_message(message):
#     message_bytes = json.dumps(message, sort_keys=True).encode()
#     signature = miner_keypair.sign(message_bytes)
#     return signature.hex()

# def send_signed_request(url, method='post', data=None):
#     signature = sign_message(data) if data else sign_message({})
#     headers = {
#         'Public-Key-Id': miner_keypair.ss58_address,
#         'Signature': signature
#     }
#     try:
#         if method.lower() == 'post':
#             response = requests.post(url, json=data, headers=headers)
#         else:
#             response = requests.get(url, headers=headers)
#         return response.json()
#     except json.JSONDecodeError:
#         # Handle cases where the response is not in JSON format
#         return {"error": "Failed to decode JSON response."}




def setup(rank, world_size, store_address, store_port, timeout=30):
    #torch.distributed.destroy_process_group()
    print(f"World Size in miner process: {world_size} @ rank {rank}")
    store = TCPStore(store_address, store_port, None,False,timedelta(seconds=timeout) )
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, store=store)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    #torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()


# class ValidateGradientHook:
#     def __init__(self, validator_urls):
#         # Ensure validator_urls is a list
#         if not isinstance(validator_urls, list):
#             raise ValueError("validator_urls must be provided as a list.")
#         self.validator_urls = validator_urls

#     def hook(self, state, bucket):
#         for tensor in bucket.gradients():
#             checksum = hashlib.md5(tensor.numpy()).hexdigest()
#             # Send to each validator URL
#             for url in self.validator_urls:
#                 requests.post(f"{url}/validate_gradient", json={"checksum": checksum})
#         return bucket.gradients()

def send_model_checksum(model, rank, validator_urls):
    # Ensure validator_urls is a list
    if not isinstance(validator_urls, list):
        raise ValueError("validator_urls must be provided as a list.")
    
    model_state = model.state_dict()
    model_bytes = json.dumps({k: v.tolist() for k, v in model_state.items()}, sort_keys=True).encode()
    checksum = hashlib.md5(model_bytes).hexdigest()
    # Send to each validator URL
    for url in validator_urls:
        requests.post(f"http://{url}/validate_model", json={"rank": rank, "checksum": checksum})

# Existing meta_miner.py code with modifications to use Bittensor wallet for signing and authentication
def create_signed_message(message):
    """Sign a message and return the signature."""
    global wallet
    signature = wallet.hotkey.sign(message).hex()  # Convert bytes to hex string for easy transmission
    public_address = wallet.hotkey.ss58_address
    return message, signature, public_address

def send_metrics(metrics, rank, validator_urls):
    timestamp = time.time()
    message, signature, public_address = create_signed_message(timestamp)
    data = {'message': message, 'signature': signature, 'public_address': public_address, "metrics": metrics, "rank": rank}
    # Ensure metrics is a dictionary
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be provided as a dictionary.")
    # Ensure validator_urls is a list
    if not isinstance(validator_urls, list):
        raise ValueError("validator_urls must be provided as a list.")
    
    # Send to each validator URL
    for url in validator_urls:
        try:
            requests.post(f"http://{url}/validate_metrics", json={"rank": rank, "checksum": checksum, "metrics": metrics})
        except:
            pass#FIXME log sth

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(144, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

def train(rank, world_size, epochs, batch_size, validator_urls, store_address, store_port):
    print("setting up")
    print(rank)
    setup(rank, world_size, store_address, store_port)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("loading dataset")

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("loading model")
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_loss = torch.zeros(2).to(LOCAL_RANK)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)
    model = Net().to(LOCAL_RANK)
    model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True))
    optimizer = optim.Adam(model.parameters())

    #validate_gradient_hook = ValidateGradientHook(validator_urls)
    # Register gradient validation hook with DDP model
    #model.register_comm_hook(None, validate_gradient_hook.hook)
    

    for epoch in range(epochs):
        print("begin training")
        #world_size = update_world_size(orchestrator_url)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        print("Loading data")
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        model.train()
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(LOCAL_RANK), target.to(LOCAL_RANK)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target, reduction='sum')
            loss.backward()
            optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
                try:
                    send_metrics({"loss": loss.item()}, rank, validator_urls)
                except Exception as e:
                    print(e)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        try:
            send_model_checksum(model, rank, validator_urls)
        except Exception as e:
            print(e)

    if rank == 0:
        torch.save(model.state_dict(), "mnist_model.pt")
    
    cleanup()

if __name__ == "__main__":
    config = Configurator.combine_configs()
    #FIXME add wallet, etc
    BittensorNetwork.initialize(config)
    wallet = BittensorNetwork.wallet
    subtensor = BittensorNetwork.subtensor
    metagraph = BittensorNetwork.metagraph

    train(rank=config.rank, world_size=config.world_size, epochs=config.epochs, batch_size=config.batch_size, validator_urls=config.validator_urls,
        store_address=config.store_address, store_port=config.store_port)
