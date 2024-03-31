

## There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.



Subnet25

## Exploring New Frontiers : א・hivemind・25
## Essential Guide for Bittensor Enthusiasts:
Beginners to Bittensor are encouraged to start with the basics by visiting the Bittensor official site. This foundational step is crucial for understanding the innovative landscape Bittensor is shaping.

## Launching Subnet Hivemind-25: Our Quest:
Our ambition is to train the largest ever Large Language Model (LLM), harnessing the unique strengths of the Bittensor network's and in a completely decentralised manner. Our approach is rooted in transparency and open collaboration in the AI space.

## The Significance of Our Endeavor:
An overview of the "LMSYS Chatbot Arena Leaderboard" highlights the predominance of proprietary models. Our endeavor seeks to challenge this norm, ushering in a new phase of open and inclusive AI development practices.

Subnet25

## What we’ve done
After a period of intense experimentation and evaluation, we have successfully trained our inaugural model (tinygpt2), marking the first ever incentivized distributed training over the internet (incentivized with TAO).

We are currently in our next project phase (training a slightly larger GPT2 model - 677m params), and we invite you to join our mission by dedicating your computational resources towards this training run. All you have to do is run a miner, and it may just change the world.

## Live Status
Done : Train TINYGPT
In Progress : GPT2 Training Run - 667m Params
Roadmap (Dropping 16 Feb 2024)
Office Hours (Timing TBC - Dropping 16 Feb 2024)
Step-by-Step Guide to Mining on Hivemind (Dropping 16 Feb 2024)
## Known Issues (fix in progress)
load_state_from_peers errors
⁠All_reduce errors
Timeout errors
suddenly diminishing miner incentive
## New Bounties
-- Improving our validation mechanism is key, take a look at it and make a propposal (@bitcurrent on discord) - if accepted you can execute. 5t reward upon exection. Please propose before execution!
## Frequently Asked Questions
-- What are the minimum requirements to run a validator? A GPU with a minimum of 16GB RAM e.g. RTX A4000
-- What are the minimum requirements to run a miner? A GPU with a minimum of 16GB RAM e.g. RTX A4000

## Running a Miner on HiveMind : A Step-by-Step Guide

## Running a Miner on Testnet
For detailed instructions on how to run a miner on the testnet, please refer to the following documentation: Running a Miner on Testnet

## Prerequisites
Before you start, ensure your system meets the following requirements:

Your machine meets the minimum hardware requirements for mining on subnet 25: miner/validator GPU - 16GB RAM e.g. RTX A4000.
You have the requisite amount of tao in your wallet for registration fees (approx. 0.00001 Tao at the time of writing).
This repository requires python3.8 or higher.

## Setting Up
Clone the Repository: Start by cloning the Distributed Training repository.

git clone https://github.com/bit-current/DistributedTraining

Navigate to the Repository: Change your directory to the cloned repository.
cd DistributedTraining

Install Dependencies: Install all necessary dependencies and run post-install scripts.
pip install -e . && python post_install.py

You also need to install pm2.
On linux:
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
On macOS:
brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update

Wandb Login: You need a Weights & Biases account for tracking runs. If you don't have one, sign up at https://wandb.ai/site and use your API key to log in.
wandb login <your_wandb_api_key>

Register on Subnet 25: To register, execute the following command:
btcli subnet register --netuid 25 --subtensor.network test --wallet.name miner --wallet.hotkey hotkey

Once you have installed this repo you can run the miner and validator with auto updates enabled using the following commands.

## To run the miner
chmod +x run_miner.sh
```pm2 start run_miner.sh --name distributed_training_miner_auto_update --
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --miner.bootstrapping_server http://35.239.40.23:4999/return_dht_address
```

## To run the validator
chmod +x run_validator.sh
```pm2 start run_validator.sh --name distributed_training_auto_update --
    --netuid <your netuid> # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url> # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>
```
## Distributed Training Framework

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
--miner.bootstrapping_server http://35.239.40.23:4999/return_dht_address
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
btcli s register --netuid 25 --subtensor.network finney
```

## Miner Run Command

```
pm2 start neurons/miner.py --interpreter python3 --name trainer -- --netuid 25 --wallet.name xxxxx --wallet.hotkey xxxxx --subtensor.network finney --logging.debug --miner.bootstrapping_server http://35.239.40.23:4999/return_dht_address
```

## Validator

### Validators need to have at least 1000 TAO to be able to set weights.

```
pm2 start neurons/validator.py --interpreter python3 --name VAL -- --netuid 25 --wallet.name xxxx --wallet.hotkey xxxx --axon.external_ip x.x.x.x --axon.port xxxx --subtensor.network finney --logging.debug --axon.ip x.x.x.x --axon.external_port xxxx --flask.host_address 0.0.0.0 --flask.host_port xxxx
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
