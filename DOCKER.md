# how to use hivetrain for docker

## install dependencies

1. [Docker](https://docs.docker.com/engine/install/)
2. [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## clone the repo
```
git clone https://github.com/bit-current/DistributedTraining
```

## move into the repo
```
cd DistributedTraining
```

## checkout the dev branch
```
git checkout test-lightning
```

## build the docker image
```
docker compose build
```

## make a .env file
Edit the existing .env file to include values that reflect your machine/network.

## join a training run
To join the existing training run on the subnet. Find the peer id of a running node on the network.
This will be provided for you. For the lastest, see pinned post on the discord channel. 
 add this environment variable to your `.env` file:
```
INITIAL_PEERS="/ip4/peer_ip/tcp/peer_dht_port/p2p/12D3KooWE_some_hash_that_looks_like_this_VqgXKo9EUQ4hguny9"
```
After that, you may join the training run with:
```
docker compose up
```

## final notes

Your machine will print your own peer ID to the console at startup. It should look like this:
```
PEER-ID: /p2p/12D3KooWF9KB7PVUdbct4ryCMzDjbNT1q2w5XMw9iVG6tisY4ThB
```
If Hivemind is under-utilizing your GPU (i.e. it's not using all of your available VRAM), you may try to increase the batch size being used. To do this, add this environment variable to your `.env` file:
```
BATCH_SIZE=2 (or 3, or whatever)
```
You will know that training is progressing when you see output like this:
```
hivetrain-1  | LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
hivetrain-1  | 
hivetrain-1  |   | Name  | Type            | Params
hivetrain-1  | ------------------------------------------
hivetrain-1  | 0 | model | GPT2LMHeadModel | 186 M 
hivetrain-1  | ------------------------------------------
hivetrain-1  | 186 M     Trainable params
hivetrain-1  | 0         Non-trainable params
hivetrain-1  | 186 M     Total params
hivetrain-1  | 747.418   Total estimated model params size (MB)
hivetrain-1  | Global Step: 0, Local Loss: 12.069, Peers: 0
hivetrain-1  | Global Step: 0, Local Loss: 12.063, Peers: 1
hivetrain-1  | Global Step: 0, Local Loss: 11.852, Peers: 2
```
