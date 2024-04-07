import torch
from huggingface_hub import Repository, HfFolder
from hivetrain.averaging_logic import Averager
# Assuming `model` is your PyTorch model, `scores` contain your keys and their respective scores,
# and `get_weights(key)` is a function to retrieve serialized gradients

args = Configurator.combine_configs()

BittensorNetwork.initialize(args)

my_hotkey = BittensorNetwork.wallet.hotkey.ss58_address
my_uid = BittensorNetwork.metagraph.hotkeys.index(my_hotkey)

address_store = ChainMultiAddressStore(BittensorNetwork.subtensor, args.netuid,BittensorNetwork.wallet)

# Create an instance of DHTManager
dht_manager = DHTManager(
    address_store=address_store,
    bittensor_network=BittensorNetwork,
    my_uid=my_uid,
    dht_host_address=args.miner.dht_host_address,
    dht_tcp_port=args.miner.dht_tcp_port,
    dht_udp_port=args.miner.dht_udp_port,
    dht_external_ip=args.miner.dht_external_ip,
    dht_private_key=args.miner.dht_private_key
)

# Use the DHTManager to manage DHT interactions
dht_manager.manage_dht()

# Define your model's local directory and repository ID
local_dir = args.averager.save_directory #TODO add me to config :)
repo_id = args.averager.hf_repository #TODO add me to config :)

# Save the updated model
model.save_pretrained(local_dir)

averager = Averager(model, local_dir, repo_id,dht,bittensor_network, hf_token=None)
averager
# Push the model to the Hugging Face Hub
push_to_hf_hub(local_dir=local_dir, repo_id=repo_id, hf_token=args.averager.hf_token, commit_message=f"Updated model SN25 with {}")#FIXME add block numbers
