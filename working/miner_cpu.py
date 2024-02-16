import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.distributed.ddp import DistributedDataParallel as FSDP
from torch.utils.data import DataLoader
import requests
import torch.distributed as dist

# def join_orchestrator(orchestrator_url):
#     response = requests.post(f"{orchestrator_url}/join").json()
#     os.environ['RANK'] = str(response['rank'])
#     os.environ['WORLD_SIZE'] = str(response['world_size'])
#     os.environ['MASTER_ADDR'] = response['master_addr']
#     os.environ['MASTER_PORT'] = response['master_port']
#     return response['rank'], response['world_size']

# def update_world_size(orchestrator_url):
#     response = requests.get(f"{orchestrator_url}/update").json()
#     new_world_size = response['world_size']
#     if new_world_size != dist.get_world_size():
#         dist.destroy_process_group()
#         torch.distributed.init_process_group("gloo", rank=dist.get_rank(), world_size=new_world_size)
#     return new_world_size

def setup(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

def join_orchestrator(orchestrator_url):
    response = requests.post(f"{orchestrator_url}/register").json()
    return response['rank'], response['world_size']

def update_world_size(orchestrator_url, rank):
    response = requests.post(f"{orchestrator_url}/update", json={"rank": rank}).json()
    return response['world_size']


class ValidateGradientHook:
    def __init__(self, validator_url):
        self.validator_url = validator_url

    @staticmethod
    def hook(state, bucket):
        for tensor in bucket.get_tensors():
            checksum = hashlib.md5(tensor.numpy()).hexdigest()
            requests.post(f"{state.validator_url}/validate_gradient", json={"checksum": checksum})
        return bucket.get_tensors()

def send_model_checksum(model, rank, validator_url):
    model_state = model.state_dict()
    model_bytes = json.dumps({k: v.tolist() for k, v in model_state.items()}, sort_keys=True).encode()
    checksum = hashlib.md5(model_bytes).hexdigest()
    requests.post(f"{validator_url}/verify_model", json={"rank": rank, "checksum": checksum})


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

def train(rank, world_size, epochs, batch_size, orchestrator_url):
    print("setting up")
    print(rank)
    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("loading dataset")
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("loading model")
    model = Net()
    model = DDP(model)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        print("begin training")
        #world_size = update_world_size(orchestrator_url)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        print("Loading data")
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target
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
    import os
    parser = argparse.ArgumentParser(description='PyTorch FSDP Example')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--orchestrator-url', type=str, required=True, help='URL of the orchestrator server')
    parser.add_argument('--world-size', type=int, required=True, help='URL of the orchestrator server')

    args = parser.parse_args()

    rank, _ = join_orchestrator(args.orchestrator_url)
    train(rank, args.world_size, args.epochs, args.batch_size, args.orchestrator_url)
