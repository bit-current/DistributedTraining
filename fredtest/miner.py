import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import requests
import os
import json
from substrateinterface import Keypair

# Assuming the use of a simple CNN for MNIST dataset
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

def receive_task(orchestrator_url):
    """Simulate receiving a dynamic task from the orchestrator."""
    response = requests.get(f"{orchestrator_url}/get_task")
    if response.status_code == 200:
        task = response.json()
        return task
    else:
        print("Failed to receive task.")
        return None

def submit_update(update, orchestrator_url):
    """Submit model updates to the orchestrator."""
    response = requests.post(f"{orchestrator_url}/submit_update", json=update)
    if response.status_code == 200:
        print("Update submitted successfully.")
    else:
        print("Failed to submit update.")

def partial_aggregate(update, orchestrator_url):
    """Participate in partial aggregation."""
    response = requests.post(f"{orchestrator_url}/partial_aggregate", json=update)
    if response.status_code == 200:
        aggregated_update = response.json()
        return aggregated_update
    else:
        print("Failed to participate in partial aggregation.")
        return None

def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training Miner")
    parser.add_argument("--orchestrator-url", type=str, required=True, help="URL of the orchestrator")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    
    while True:
        task = receive_task(args.orchestrator_url)
        if task is None:
            continue

        dataset = datasets.MNIST('../data', train=True, download=True, 
                                 transform=transforms.Compose([transforms.ToTensor(), 
                                                               transforms.Normalize((0.1307,), (0.3081,))]))
        # Assuming task contains indices for the data subset
        train_subset = Subset(dataset, indices=task['indices'])
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

        train(model, device, train_loader, optimizer)

        # Assuming the task requires submitting model updates
        if task.get('submit_update', False):
            model_update = {"weights": model.state_dict()}  # Simplification for demonstration
            submit_update(model_update, args.orchestrator_url)

        # Assuming the task may require participation in partial aggregation
        if task.get('participate_in_aggregation', False):
            aggregated_update = partial_aggregate(model_update, args.orchestrator_url)
            if aggregated_update:
                model.load_state_dict(aggregated_update['weights'])

if __name__ == "__main__":
    main()
