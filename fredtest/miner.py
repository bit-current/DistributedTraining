import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import requests
import torch.distributed as dist
import json
import os
from substrateinterface import Keypair

# Define your neural network architecture
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
        return torch.log_softmax(x, dim=1)

# Functions for digital signing and communication with orchestrator
def sign_message(message, keypair):
    message_bytes = json.dumps(message, sort_keys=True).encode()
    signature = keypair.sign(message_bytes)
    return signature.hex()

def send_signed_request(url, data, keypair):
    signature = sign_message(data if data else {}, keypair)
    headers = {'Public-Key-Id': keypair.ss58_address, 'Signature': signature}
    response = requests.post(url, json=data, headers=headers)
    return response.json() if response.ok else None

def join_orchestrator(orchestrator_url, keypair):
    data = {"miner_address": keypair.ss58_address}
    response = send_signed_request(f"{orchestrator_url}/register_miner", data, keypair)
    if response and 'rank' in response and 'world_size' in response:
        return response['rank'], response['world_size']
    print("Failed to join orchestrator.")
    return None, None

def update_world_size(orchestrator_url, rank, keypair):
    data = {"rank": rank}
    response = send_signed_request(f"{orchestrator_url}/update_world_size", data, keypair)
    return response['world_size'] if response else None

def main(args):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    
    rank, world_size = join_orchestrator(args.orchestrator_url, keypair)
    if rank is None or world_size is None:
        print("Registration with the orchestrator failed. Exiting.")
        return

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = Net().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        for _, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "mnist_model.pt")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--orchestrator-url", type=str, required=True, help="URL of the orchestrator server")
    args = parser.parse_args()

    main(args)
