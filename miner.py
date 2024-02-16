import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
import requests

def join_orchestrator(orchestrator_url):
    response = requests.post(f"{orchestrator_url}/join").json()
    return response['rank'], response['world_size']

def update_world_size(orchestrator_url):
    response = requests.get(f"{orchestrator_url}/update").json()
    return response['world_size']

def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define your model architecture here
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
    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    model = Net().to(rank)
    model = FSDP(model.cuda())
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        # Update world size at the beginning of each epoch
        world_size = update_world_size(orchestrator_url)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
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
