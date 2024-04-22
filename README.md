> There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.

# Distributed Training Framework

(WARNING: IN ACTIVE DEVELOPMENT)

## Introduction

This project introduces a cutting-edge approach to distributed deep learning, utilizing the Bittensor network. Our method incentivizes participants by rewarding the generation of optimal weights that contribute significantly to minimizing the overall loss of the base model.

To streamline the process and reduce communication overhead between miners, we integrate Hugging Face as a central hub. This serves as an intermediary, facilitating efficient miner-
validator communications without the complexities of direct exchanges.

Key Components: 
* Miners: Miners are responsible for training the language model. They compute the weight delta—the difference between the weights of the trained model and the base model. This delta is then uploaded to Hugging Face, from where it can be accessed by validators. 
* Validators: Validators play a crucial role in assessing the miners' contributions. They download the weight deltas from Hugging Face and evaluate them based on their impact on the model’s performance, focusing on metrics such as loss reduction and accuracy. 
* Averager: We also introduce an averager, which analyzes the combined effect of individual weight contributions to determine the optimal combination that results in the lowest possible loss.

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
git checkout chain_meta
```

## Make a .env File

Edit the existing .env file to include values that reflect your machine/network.

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
