
<div align="center">

# **Distributed Training Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

---

![Subnet25](assets/Subnet25.jpg)

# Exploring New Frontiers : א・hivemind・25

## Essential Guide for Bittensor Enthusiasts:
Beginners to Bittensor are encouraged to start with the basics by visiting the [Bittensor official site](https://www.bittensor.com). This foundational step is crucial for understanding the innovative landscape Bittensor is shaping.

## Launching Subnet Hivemind-25: Our Quest:
Our ambition is to train the largest ever Large Language Model (LLM), harnessing the unique strengths of the Bittensor network's and in a completely decentralised manner. Our approach is rooted in transparency and open collaboration in the AI space. 
## The Significance of Our Endeavor:
An overview of the "LMSYS Chatbot Arena Leaderboard" highlights the predominance of proprietary models. Our endeavor seeks to challenge this norm, ushering in a new phase of open and inclusive AI development practices.

![Subnet25](assets/llmscoreboard.webp)


## What we’ve done 
After a period of intense experimentation and evaluation, we have successfully trained our inaugural model (tinygpt2), marking the first ever incentivized distributed training over the internet (incentivized with TAO). 

We are currently in our next project phase  (training a slightly larger GPT2 model - 677m params), and we invite you to join our mission by dedicating your computational resources towards this training run. All you have to do is run a miner, and it may just change the world.

## Live Status 
* Done : Train TINYGPT  
* In Progress : GPT2 Training Run - 667m Params
* Roadmap (Dropping 16 Feb 2024)
* Office Hours (Timing TBC - Dropping 16 Feb 2024)
* Step-by-Step Guide to Mining on Hivemind (Dropping 16 Feb 2024)


## Known Issues (fix in progress)
* load_state_from_peers errors
* ⁠All_reduce errors
* Timeout errors
* suddenly diminishing miner incentive

## Frequently Asked Questions
* What are the minimum requirements to run a validator? A GPU with a minimum of 16GB RAM e.g. RTX A4000
* What are the minimum requirements to run a miner? A GPU with a minimum of 16GB RAM e.g. RTX A4000



# Getting Started - Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.

1. Install this repository
```bash
git clone https://github.com/bit-current/DistributedTraining
cd DistributedTraining
pip install -e . && python post_install.py
```

2. Log in to wandb:
```bash
wandb login <your_wandb_api_key>
```

3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.

**On Linux**:
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
``` 
**On Mac OS**
```bash
brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
```
---

Once you have installed this repo you can run the miner and validator with **auto updates enabled** using the following commands.
```bash
# To run the miner
chmod +x run_miner.sh
pm2 start run_miner.sh --name distributed_training_miner_auto_update --
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>

# To run the validator
chmod +x run_validator.sh
pm2 start run_validator.sh --name distributed_training_auto_update --
    --netuid <your netuid> # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url> # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>
```

</div>

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
