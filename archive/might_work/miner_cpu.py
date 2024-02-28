import argparse
import os
import hashlib
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
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2 = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.dropout2(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #rank = os.environ["RANK"]
    #world_size = os.environ["WORLD_SIZE"]
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs, batch_size, validator_urls):
    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = Net()#.cuda(rank)
    model = DDP(model)#, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1):
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            #data, target = data.cuda(rank), target.cuda(rank)
            data, target = data, target
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)], Batch {batch_idx}, Loss {loss.item()}")
                send_metrics({"loss": loss.item()}, rank, validator_urls)
        send_model_checksum(model, rank, validator_urls)
    
    if rank == 0:
        torch.save(model.state_dict(), "mnist_model.pt")

    cleanup()

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    #world_size = int(os.environ['WORLD_SIZE'])

    parser = argparse.ArgumentParser(description='PyTorch FSDP Example')
    parser.add_argument("--world-size", type=int, default=3, help="World size for the distributed training")
    parser.add_argument("--batch-size", type=int, default=1, help="Size of batch per forward/backward pass")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--validator-urls', type=str,nargs="+", help='URLs of the test validators')#FIXME add the main from btt

    args = parser.parse_args()

    train(rank, args.world_size, args.epochs, args.batch_size, args.validator_urls)
