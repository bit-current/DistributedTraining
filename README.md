> There is no passion to be found playing small - in settling for a life that is less than the one you are capable of living. Nelson Mandela.

Alone, we may be small. Together - we can do great things.

![image](https://github.com/bit-current/NewArchScrapBook/assets/7602667/61c20a8e-22a9-4636-91dc-96152d70a0de)


# Distributed PyTorch Mining Framework

## Vision
This project aims to revolutionize distributed computing and mining by leveraging PyTorch and the Bittensor blockchain. It introduces a scalable, efficient, and secure system for training deep learning models across distributed nodes. Participants are rewarded in TAO cryptocurrency for their computational contributions, fostering a decentralized ecosystem of computational resources.

## Usefulness
- **Decentralized Computing:** Offers a framework for executing deep learning tasks across distributed nodes, reducing reliance on centralized cloud services.
- **Incentivization Scheme:** Integrates with Bittensor to reward miners and meta-miners based on their computational contributions and the validity of their work.
- **Flexibility and Scalability:** Easily scales with the addition of new nodes, suitable for a wide range of computational demands.

## How to Use
1. **Configure Miners:** Run `miner.py` on each participating node, specifying rank, world size, and validator URLs.
2. **Launch Orchestrator:** Start `orchestrator_simple_auth.py` to manage mining processes, including registration and task distribution.
3. **Meta Miner Operations:** Utilize `meta_miner.py` for higher-level management, allowing miners to run and crash without needing manual resets.
4. **Validator Script:** Engage the validator script to monitor statistical anomalies in loss and ensure model training integrity.

### Example Command

#### Orchestrator
Run the orchestrator script with the following command, replacing placeholders with your actual values where needed. This script initializes the orchestrator with specific network and TCP store settings.

```shell
python orchestrator_simple_auth.py --host-address [orchestrator_host_ip] --port [orchestrator_port] --subtensor.chain_endpoint [blockchain_ws_endpoint] --tcp-store-port [tcp_store_port] --tcp-store-address [tcp_store_ip]
```

- **[orchestrator_host_ip]:** The IP address the orchestrator service listens on. Use `0.0.0.0` for all IPv4 addresses on the local machine.
- **[orchestrator_port]:** The port number for the orchestrator service. Example: `5000`.
- **[blockchain_ws_endpoint]:** WebSocket endpoint for the blockchain. Example: `ws://127.0.0.1:9946` for a local blockchain node.
- **[tcp_store_port]:** Port number for the TCP store service. Use a unique port number, such as `2001`.
- **[tcp_store_ip]:** IP address for the TCP store. Typically `127.0.0.1` for local setups.

#### Meta Miner
The meta miner script is used to manage miner processes and interact with the blockchain. Use the command below, adjusting parameters as needed.

```shell
python meta_miner.py --netuid [network_uid] --orchestrator-url [orchestrator_url] --batch-size [batch_size] --epochs [epochs] --miner-script [miner_script_name] --subtensor.chain_endpoint [blockchain_ws_endpoint] --wallet.name [wallet_name] --wallet.hotkey [wallet_hotkey] --host-address [meta_miner_host_ip] --port [meta_miner_port] --neuron.vpermit_tao_limit [vpermit_tao_limit] --axon.port [axon_port] --axon.ip [masked_ip] --axon.external_port [axon_external_port] --axon.external_ip [masked_external_ip] --tcp-store-port [tcp_store_port] --tcp-store-address [tcp_store_ip]
```

- All parameters should be configured according to your specific setup. Note that `[axon.ip]` and `[axon.external_ip]` have been masked and should be set to the appropriate IP addresses based on your network configuration. These can be the same if the service is directly accessible or different if translated through NAT or a proxy.
- The `[tcp_store_port]` and `[tcp_store_address]` should match those used in the orchestrator command for proper communication.

#### Validator
The validator script monitors and validates the integrity of the mining process. Execute the validator with the following parameters:

```shell
python validator.py --netuid [network_uid] --orchestrator-url [orchestrator_url] --batch-size [batch_size] --epochs [epochs] --miner-script [miner_script_name] --subtensor.chain_endpoint [blockchain_ws_endpoint] --wallet.name [validator_wallet_name] --wallet.hotkey [validator_wallet_hotkey] --host-address [validator_host_ip] --port [validator_port] --neuron.vpermit_tao_limit [vpermit_tao_limit] --axon.port [validator_axon_port] --axon.ip [masked_ip] --axon.external_port [validator_external_port] --axon.external_ip [masked_external_ip]
```

- Ensure the `[axon.ip]` and `[axon.external_ip]` settings are correctly configured for the validator, similar to the meta miner setup. These IP addresses are masked for privacy and should be set according to your network's requirements.
- The validator's `[host-address]` and `[port]` must be distinct from those used by the orchestrator and meta miner to avoid conflicts.

Remember to replace placeholder values with actual configurations relevant to your setup. 

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


### Updated Sections Based on Clarifications:
- **Vision:** Added mention of Bittensor and TAO cryptocurrency.
- **Meta Miner Operations:** Clarified the role of meta-miners in enhancing the resilience of the mining network.
- **Communication and Support:** Added mention of the project and Bittensor Discord channels for community engagement.
- **Validator Script:** Brief description of its functionality related to anomaly detection and model training integrity.
- **Contributing:** Encouraged testing and breaking the system to improve resilience.

### Remaining Actions:
- Add actual links to the Discord channels in the **Communication and Support** section.
- Finalize and review the README for any project-specific details that may have been overlooked.

This README provides a comprehensive overview of your project, its goals, and how users can engage with it. Let me know if there's anything else you'd like to include or modify!
