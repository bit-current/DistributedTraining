import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
import requests
import torch.distributed as dist
import json
import os
from substrateinterface import Keypair
import hashlib

# Assuming the Keypair for the miner is generated or loaded here
miner_keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

def sign_message(message):
    message_bytes = json.dumps(message, sort_keys=True).encode()
    signature = miner_keypair.sign(message_bytes)
    return signature.hex()

def send_signed_request(url, method='post', data=None):
    signature = sign_message(data if data else {})
    headers = {
        'Public-Key-Id': miner_keypair.ss58_address,
        'Signature': signature
    }
    if method.lower() == 'post':
        response = requests.post(url, json=data, headers=headers)
    else:
        response = requests.get(url, json=data, headers=headers)  # Adjusted to support GET with JSON body if necessary
    return response.json()

def join_orchestrator(orchestrator_url):
    response = send_signed_request(f"{orchestrator_url}/register_miner", data={"miner_address": miner_keypair.ss58_address})
    if 'rank' in response and 'world_size' in response:
        os.environ['RANK'] = str(response['rank'])
        os.environ['WORLD_SIZE'] = str(response['world_size'])
        return response['rank'], response['world_size']
    else:
        print("Failed to join orchestrator.")
        return None, None

def update_world_size(orchestrator_url, rank):
    response = send_signed_request(f"{orchestrator_url}/update_world_size", data={"rank": rank})
    if 'world_size' in response:
        return response['world_size']
    else:
        print("Failed to update world size.")
        return None

def setup(rank, world_size):
    if rank is not None and world_size is not None:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

def train(rank, world_size, epochs, batch_size, orchestrator_url):
    if rank is None or world_size is None:
        return  # Early exit if registration failed

    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    model = Net().to(rank)
    model = FSDP(model)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        world_size = update_world_size(orchestrator_url, rank)  # Update world size at the beginning of each epoch
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    if rank == 0:
        torch.save(model.state_dict(), "mnist_model.pt")
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FSDP Example')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--orchestrator-url', type=str, required=True, help='URL of the orchestrator server')

    args = parser.parse_args()

    rank, world_size = join_orchestrator(args.orchestrator_url)
    train(rank, world_size, args.epochs, args.batch_size, args.orchestrator_url)
