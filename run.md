 # Neurons Miner Shell Script

This shell script is designed to execute the `neurons/miner.py` Python script using the specified arguments. The following documentation outlines the logic of the script.

```bash
#!/bin/bash

# Commence the execution of the neurons miner script with the provided arguments
python3 neurons/miner.py \
  --netuid 25 \
  --wallet.name "JJcold" \
  --wallet.hotkey "JJb" \
  --miner.bootstrapping_server "http://35.239.40.23:4999/return_dht_address"
```

## Script Logic

1. The shebang `#!/bin/bash` line specifies that this script should be executed using the Bash shell interpreter.

2. The command `python3 neurons/miner.py` invokes the Python 3 interpreter and runs the `neurons/miner.py` script located within the current directory.

3. The following arguments are passed to the `neurons/miner.py` script:
   - `--netuid 25`: Sets the network ID to 25. This value may represent a specific Monero obfuscated pool or a different network configuration.
   - `--wallet.name "JJcold"`: Specifies the name of the cold wallet as "JJcold." The miner will use this wallet for storing and managing monero rewards.
   - `--wallet.hotkey "JJb"`: Sets the hot key (recovery phrase seed) for the cold wallet to "JJb." This seed is required to access the funds in the cold wallet.
   - `--miner.bootstrapping_server "http://35.239.40.23:4999/return_dht_address"`: Configures the miner to connect to a bootstrapping server located at `http://35.239.40.23:4999/return_dht_address`. This server will provide the miner with the necessary information to connect to other nodes on the Monero network and begin mining.