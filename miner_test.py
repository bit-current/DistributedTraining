
import torch
import os
from torch import nn
import torch.nn.functional as F
import time
import hivemind

# Create dataset and model, same as in the basic tutorial
model = nn.Linear(2, 3)
device = "cpu"

opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(initial_peers=["/ip4/54.80.217.105/tcp/8009/p2p/12D3KooWHfQMoqVqCqFMUNt8Dy24kfNu1PTuTLCahW4swDbpb7Zi"], 
                   start=True)

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=32,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=200,  # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,            # wrap the SGD optimizer defined above
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
    verbose=True              # print logs incessently
)


features = torch.randn(100, 2)
targets = features @ torch.randn(2, 3)
batch_size = 1

while True:
    opt.load_state_from_peers()
    print("weights pre:", model.weight.data[...])

    batch = torch.randint(0, len(features), (batch_size,))
    loss = F.mse_loss(model(features[batch]), targets[batch])
    loss.backward()
    opt.step()
    opt.zero_grad()

    print(opt.state_averager.is_alive())
    print(opt.tracker.is_alive())
    print(opt.grad_averager)
    print("weights post:", model.weight.data[...])
    time.sleep(5)