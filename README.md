> There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.

# Distributed Training Framework

(WARNING: IN ACTIVE DEVELOPMENT)

## Introduction

This project introduces a cutting-edge approach to distributed deep learning, utilizing the Bittensor network. Our method incentivizes participants by rewarding the generation of optimal weights that contribute significantly to minimizing the overall loss of the base model.

To streamline the process and reduce communication overhead between miners, we integrate Hugging Face as a central hub. This serves as an intermediary, facilitating efficient miner-validator communications without the complexities of direct exchanges.

Key Components: 
* Miners: Miners are responsible for training a model. Each miner trains a weight-delta. A weight-delta is the difference between the weights of the trained model and the base model. This delta is then uploaded to Hugging Face, from where it can be accessed by validators. 
* Validators: Validators asses the loss reduction by each miner on a randomized test set. They download the weight deltas from Hugging Face and evaluate them based on their impact on the model’s performance, focusing on metrics such as loss reduction and accuracy.Better performing miners that improve on the base model are assigned better scores.
* Averager: We also introduce an averager node, which is a centralized node run by the subnet owner. The averager is responsible for providing the averaged model that becomes the base model for miners and validators, this is repeated every averaging interval. The averager performs a weighted average of the parameters resulting in an averaged model. Currently the weights of the weighted average are also parameterized allowing the process to be optimized to find the best averaged model. 

## Clone the Repo

```
git clone https://github.com/bit-current/DistributedTraining
```

## Move into the Repo

```
cd DistributedTraining
```

## Remove Previous Hivetrain installation

```
pip uninstall hivetrain
```

## Install Repo + Requirements

```
pip install -e .
```

## Hugging Face
Continue setting up by following these step:

### 1. Create a Hugging Face Account
If you don't already have a Hugging Face account, you'll need to create one:

Visit [Hugging Face](https://huggingface.co/) to sign up
### 2. Create a Hugging Face Model Repository
Once you have your Hugging Face account, you need to create a model repository:
* Navigate to your profile by clicking on your username in the top right corner.
* Click on "New Model" (you may find this button under the "Models" section if you have existing models).
* Fill in the repository name, description, and set the visibility to public.
* Click on "Create Model" to establish your new model repository.
### 3. Generate a Token for the Repository
To allow programmatic communication with your repository, you will need to generate an authentication token:

* From your Hugging Face account, go to "Settings" by clicking on your profile icon.
* Select the "Access Tokens" tab from the sidebar.
* Click on "New Token".
* Name your token and select the "write" access to be able to upload changes.
* Click on "Create Token".

### 4. Create a New .env File to Store Your Hugging Face Token
Store your new token in the .env file in DistributedTranining directory:
```
HF_TOKEN="your_huggingface_token_here"
```

## Load Wallets and Register to Subnet

```
btcli regen_coldkey --mnemonic your super secret mnemonic
btcli regen_hotkey --mnemonic your super secret mnemonic
btcli s register --netuid 100 --subtensor.network test
```

## New arguments
storage.averaged_model_repo_id: The repo that is used by the averager. Currently this is Hivetrain/averaging_run_1. Changes with each training run, review changes on the discord channel .
storage.my_repo_id: Repo id for the repo that is used by a **miner only** to upload the miner's trained model weight delta 

## Miner Run Command

```
python miner.py --netuid 25 --wallet.name wallet_name --wallet.hotkey hotkey_name --storage.my_repo_id your_/your_repo --storage.averaged_model_repo_id Hivetrain/averaging_run_1
```

## Validator

### Validators need to have at least 1000 TAO to set weights on the main net and 10 TAO on the test net

```
python validator.py --netuid 25 --wallet.name wallet_name --wallet.hotkey hotkey_name --storage.averaged_model_repo_id Hivetrain/averaging_run_1
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
