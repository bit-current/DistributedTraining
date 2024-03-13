
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

## Live Status (Friday 16th Feb, 2024)
Done : Train TINYGPT  
* In Progress : GPT2 Training Run - 667m Params
* Roadmap
    * **Roadmap.Now** : Upgrade/stabilise subnet architecture
        * Outcomes : Improves all aspects of subnet operations, validaton, mining experience and stability. Much more details to come.  
    * **Roadmap.Next** : Train even larger model (still deciding which one)
    * **Roadmap.Later** : Fintune and Serve LLM on bittensor
* Office Hours (Timing TBC - Pending Architecture Upgrade)
    * **Live Guided Mining from Scratch on Hivemind (S25)** 
* Step-by-Step Guide to Mining on Hivemind (Dropping 16 Feb 2024) - **See below**
    * : See [Running a Miner on Testnet](https://github.com/bit-current/DistributedTraining/edit/main/docs/running_25_on_testnet.md).


## Known Issues (fix in progress)
* suddenly diminishing miner incentive - (fixed pushed Thursday 15th Feb - 7:30EST - We need your feedback on this one)
* load_state_from_peers errors (Should be fixed - let us know if you still see it, we're not 100% sure on this one)
* Timeout errors - on the list (not occurred recently, but let us know if it shows up again!)
* All_reduce errors - on the list

## Frequently Asked Questions
* **Q**:What are the minimum requirements to run a validator?
   * **A** GPU with a minimum of 16GB RAM e.g. RTX A4000
* **Q**: What are the minimum requirements to run a miner?
   * **A**: GPU with a minimum of 16GB RAM e.g. RTX A4000
* **Q**: I am running a miner and I see this: ERROR: Could not retrieve update: caught TypeError("StringParseError.init() missing 1 required positional argument: 'string'"), What am I doing wrong?
   * **A**: Check your port and IP configuration. If you are using runpod, refer to this: https://docs.runpod.io/pods/configuration/expose-ports
* **Q**: I am running a miner and I see this: ERROR: Attempt 2 to init DHT using initial_peer as /ip4/XX.XXX.XXX.XX, What am I doing wrong?
   * **A**: Make sure you pull the latest changes from the remote repository. ```git pull``` Sometimes it can take several hundred attemps before connecting with peers. Eventually you should see in the logs "INFO: Step 1 Loss: 6.2343.."

## How Miners are Rewarded

Hivetrain uses a simple score assignment system designed to reward users for their participation and adherence to network guidelines. The system evaluates two critical aspects of user behavior: responsiveness and loss values. By applying a set of predefined rules, we aim to foster a healthy and productive network environment where all participants are incentivized to contribute positively. Whilst maintaining network integrity with few gameable variables.

### 1.0 
Users who actively respond to network activities and maintain their losses within an acceptable threshold are awarded a score of 1.0. This top score reflects exemplary user behavior and strict adherence to network standards, highlighting the user as a model participant.
Responsive Users with Unacceptable Loss:

### 0.3 
Active users whose losses exceed the acceptable limits receive a reduced score of 0.3. While these users are engaged, the score signifies the need for improvement in managing their network activities to align with expected performance metrics.
Non-responsive Users:

### 0.1 
Users who fail to respond to network prompts or activities are assigned a baseline score of 0.1. This score acts as a safety mechanism, ensuring that users are not severely penalized for inactivity during periods of network-wide disruptions. It is a minimal support to maintain user engagement, even when external factors affect their ability to participate.

### Pseudocode 
```
for uid in uids:

  responded = has_uid_responded(uid)
  loss_acceptable = is_uid_within_acceptable_loss_value(uid, loss_responses)

  if responded:
    if loss_acceptable:
      score = 1.0
    else:
      score = 0.3
    else:
    score = 0.1
# baseline reward to keep miners afloat if there is a network wide issue.
# leeches planning on sticking to this should be deregged by honest miners when network is up.

scores[uid] = score
```

# Running a Miner on HiveMind : A Step-by-Step Guide 

## Running a Miner on Testnet
For detailed instructions on how to run a miner on the testnet, please refer to the following documentation:
[Running a Miner on Testnet](https://github.com/bit-current/DistributedTraining/edit/main/docs/running_25_on_testnet.md)

## Running a Miner on Hivemind
### Prerequisites
Before you start, ensure your system meets the following requirements:

* Your machine meets the minimum hardware requirements for mining on subnet 25: miner/validator GPU - 16GB RAM e.g. RTX A4000.
* You have the requisite amount of tao in your wallet for registration fees (approx. 0.00001 Tao at the time of writing).
* This repository requires python3.8 or higher.

### Setting Up
* Clone the Repository: Start by cloning the Distributed Training repository.  

    ```git clone https://github.com/bit-current/DistributedTraining```  
* Navigate to the Repository: Change your directory to the cloned repository.  
    ```cd DistributedTraining```  
* Install Dependencies: Install all necessary dependencies and run post-install scripts.        
    ```pip install -e . && python post_install.py```    
 * You also need to install pm2.  
 On linux:  
    ```sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update```  
 On macOS:   
    ```brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update```
* Wandb Login: You need a Weights & Biases account for tracking runs. If you don't have one, sign up at https://wandb.ai/site and use your API key to log in.  
    ```wandb login <your_wandb_api_key>```
* Register on Subnet 25: To register, execute the following command:    
```btcli subnet register --netuid 25 --wallet.name miner --wallet.hotkey hotkey```

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
