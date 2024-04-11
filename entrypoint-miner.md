 # Hive Mining Script

This script is designed to execute the HiveOS miner using `hiveminer.py` from HiveOS, a popular open-source mining platform for Monero (XMR) and other cryptocurrencies.

## Prerequisites

1. Make sure you have Python3 installed on your system.
2. Install HiveOS by following the official installation guide: [HiveOS Installation Guide](https://docs.hiveos.farm/install/)

## Usage

```bash
#!/bin/bash

python3 hivetrain/hiveminer.py \
    --initial_peers ${INITIAL_PEERS} \
    --batch_size ${BATCH_SIZE} \
    --save_every ${SAVE_EVERY}
```

The script includes the following command which launches HiveOS miner with customized options:

```bash
python3 hivetrain/hiveminer.py [options]
```

## Options

- `--initial_peers <peer1>:<port>,<peer2>:<port>,...`: A comma-separated list of initial peers to connect to the HiveOS network.
- `--batch_size <integer>`: The batch size for mining, which is a measure of how many transactions can be processed in one round before sending them off to the blockchain.
- `--save_every <number_of_blocks>`: Saves the current miner state every `<number_of_blocks>` blocks mined. This option helps maintain the miner's progress and settings across system restarts or crashes.

## Running the Script

1. Save the script in a file, let's call it `mine.sh`.
2. Make sure to replace `${INITIAL_PEERS}`, `${BATCH_SIZE}`, and `${SAVE_EVERY}` with the desired values for your mining setup.
3. Run the script using: `bash mine.sh`

This script is just a simple wrapper around the HiveOS miner, providing an easy way to launch it with custom options from the command line.