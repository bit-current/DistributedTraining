> There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.

# Distributed Mining Framework 
(WARNING : IN ACTIVE DEVELOPMENT)

## Vision
This project aims to revolutionize distributed computing and mining by leveraging distributed training on the Bittensor blockchain. It introduces a scalable, efficient, and secure system for training deep learning models across distributed nodes. Participants are rewarded in TAO cryptocurrency for their computational contributions, fostering a decentralized ecosystem of computational resources.

## Usefulness
- **Decentralized Computing:** Offers a framework for executing deep learning tasks across distributed nodes, reducing reliance on centralized cloud services.

## How to Use (Docker)

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

## install repo+requirements
```
pip install -e .
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

## How to use (No docker)

## clone the repo
```
git clone https://github.com/bit-current/DistributedTraining
```

## move into the repo
```
cd DistributedTraining
```

## install repo+requirements
```
pip install -e .
```

## Miner run command:
python hiveminer.py --netuid 100 --wallet.name some_wallet --wallet.hotkey some_hotkey --initial_peers (please add an existing miner's dht address here. Check discord pinned post)

## Validator: 
python validator.py --netuid 100 --wallet.name some_wallet_2 --wallet.hotkey some_hotkey_2 --port some_port --axon.external_ip your_valis_ip --axon.port same_some_port


## Bug Reporting and Contributions
- **Reporting Issues:** Use the GitHub Issues tab to report bugs, providing detailed steps to reproduce along with relevant logs or error messages.
- **Contributing:** Contributions are welcome! Fork the repo, make changes, and submit a pull request. Break it in as many ways as possible to help make the system resilient.

## Communication and Support
- Join our [Project Discord](#) and the [Bittensor Discord](#) to discuss the project, seek help, and collaborate with the community.

## License
Licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to the PyTorch team for their deep learning library.
- Gratitude to Bittensor for enabling decentralized computing and finance with TAO rewards.
