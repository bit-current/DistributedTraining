> There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.

# Distributed Training Framework

(WARNING: IN ACTIVE DEVELOPMENT)

## Vision

This project aims to revolutionize distributed computing and mining by leveraging distributed training on Bittensor. It introduces a scalable system for training deep learning models across distributed nodes. Participants are rewarded in TAO for their computational contributions, fostering a decentralized ecosystem of computational resources.

## How to Use (Docker - Not yet production ready)

## Install Dependencies

1. [Docker](https://docs.docker.com/engine/install/)
2. [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Clone the Repo

```
git clone https://github.com/bit-current/DistributedTraining
```

## Move into the Repo

```
cd DistributedTraining
```

## Install Repo + Requirements

```
pip install -e .
```

## Checkout the Dev Branch

```
git checkout test-lightning
```

## Build the Docker Image

```
docker compose build
```

## Make a .env File

Edit the existing .env file to include values that reflect your machine/network.

## Join a Training Run

To join the existing training run on the subnet, find the peer ID of a running node on the network.

This will be provided for you. For the latest, see the pinned post on the Discord channel.

Add this environment variable to your `.env` file:

```
INITIAL_PEERS="/ip4/peer_ip/tcp/peer_dht_port/p2p/12D3KooWE_some_hash_that_looks_like_this_VqgXKo9EUQ4hguny9"
```

After that, you may join the training run with:

```
docker compose up
```

## How to Use (No Docker)

## Clone the Repo

```
git clone https://github.com/bit-current/DistributedTraining
```

## Move into the Repo

```
cd DistributedTraining
```

## Checkout Current Branch

```
cd DistributedTraining
```

## Install Repo + Requirements

```
pip install -e .
```

## Load Wallets and Register to Subnet

```
btcli regen_coldkey --mnemonic your super secret mnemonic
btcli regen_hotkey --mnemonic your super secret mnemonic
btcli s register --netuid 100 --subtensor.network test
```

## Miner Run Command

```
python miner.py --netuid 25 --wallet.name some_test_wallet_cold --wallet.hotkey some_test_wallet_hot --initial_peers (please add an existing miner's dht address here. Check discord pinned post or ask on discord channel)
```

## Validator

### Validators need to have at least 10 test TAO to be able to set weights.

```
python validator.py --netuid 25 --wallet.name some_test_wallet_cold --wallet.hotkey some_test_wallet_hot --axon.external_ip your_external_ip --axon.port your_external_port --logging.debug --logging.trace --axon.ip your_extrenal_ip_still --axon.external_port your_external_port_still --flask.host_address on_device_ip_to_bind_to --flask.host_port on_device_port_to_bind_to
```

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
