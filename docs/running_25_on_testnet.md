# Running a Miner on Testnet: A Step-by-Step Guide
Welcome to the step-by-step guide for running a miner/validator on the testnet. This guide is designed to help you set up and test on subnet 25.

### Prerequisites
Before you start, ensure your system meets the following requirements:

* Your machine meets the minimum hardware requirements for mining on subnet 25: miner/validator GPU - 16GB RAM e.g. RTX A4000.
* You have a small amount of test tao in your test wallet for registration fees (approx. 0.00001 Tao at the time of writing).  

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

### Registration
Check Registration Cost: In order to 'mine' you need to register on the subnet (testnet). Remember registration may cost you Tao but this is just test tao which can be created. Ask on the bittensor discord

Register on Subnet 25: To register, execute the following command:    

```btcli subnet register --netuid 25 --subtensor.network test --wallet.name miner --wallet.hotkey hotkey```

### Running a Miner
To start mining on the testnet, follow these steps:

Use the following command to start the miner. Replace placeholders with your actual data.

```
chmod +x run_miner.sh \
pm2 start run_miner.sh --name distributed_training_miner_auto_update --
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.network test
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>
```
### Running a Validator

Use the following command to start the validator. Replace placeholders with your actual data.
```
chmod +x run_validator.sh
pm2 start run_validator.sh --name distributed_training_auto_update --
    --netuid 25 # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.network test
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>
```



