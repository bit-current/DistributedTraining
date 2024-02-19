import argparse
import hashlib
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

import json
import requests
import os
from substrateinterface import Keypair

# Assuming the Keypair for the miner is generated or loaded here
miner_keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

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

def join_orchestrator(orchestrator_url):
    #response = send_signed_request(f"{orchestrator_url}/register", data={})
    response = requests.post(f"{orchestrator_url}/register", data={}).json()

    os.environ['RANK'] = str(response['rank'])
    os.environ['WORLD_SIZE'] = str(response['world_size'])
    return response['rank'], response['world_size']

def update_world_size(orchestrator_url, rank):
    #response = send_signed_request(f"{orchestrator_url}/update", data={"rank": rank})
    response = requests.post(f"{orchestrator_url}/update", data={"rank": rank})
    return response['world_size']

def setup(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
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
        requests.post(f"{url}/validate_model", json={"rank": rank, "checksum": checksum})

def send_metrics(metrics, rank, validator_urls):
    # Ensure metrics is a dictionary
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be provided as a dictionary.")
    # Ensure validator_urls is a list
    if not isinstance(validator_urls, list):
        raise ValueError("validator_urls must be provided as a list.")
    
    metrics_data = json.dumps(metrics, sort_keys=True, ensure_ascii=True)
    checksum = hashlib.md5(metrics_data.encode()).hexdigest()
    # Send to each validator URL
    for url in validator_urls:
        requests.post(f"{url}/validate_metrics", json={"rank": rank, "checksum": checksum, "metrics": metrics})

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

def train(rank, world_size, epochs, batch_size, orchestrator_url, validator_urls):
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
            data, target = data, target
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
                send_metrics({"loss": loss.item()}, rank, validator_urls)

        send_model_checksum(model, rank, validator_urls)

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
    parser.add_argument('--validator-urls', type=str,nargs="+", help='URLs of the test validators')#FIXME add the main from btt

    args = parser.parse_args()

    rank, _ = join_orchestrator(args.orchestrator_url)
    train(rank, args.world_size, args.epochs, args.batch_size, args.orchestrator_url, args.validator_urls)
