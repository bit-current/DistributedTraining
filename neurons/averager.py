import torch
from huggingface_hub import Repository, HfFolder

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



# def average_gradients(scored_gradients):
#     total_score = sum(score for _, score in scored_gradients.values())
#     averaged_gradients = {name: torch.zeros_like(grad) for name, (grad, _) in scored_gradients.items()[0][1][0].items()}
    
#     if total_score > 0:
#         for key, (gradients, score) in scored_gradients.items():
#             weight = score / total_score
#             for name, grad in gradients.items():
#                 averaged_gradients[name] += grad * weight
    
#     return averaged_gradients

# def apply_averaged_gradients(model, averaged_gradients):
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in averaged_gradients:
#                 param -= averaged_gradients[name]

# def push_to_hf_hub(local_dir, repo_id, hf_token=None, commit_message="Pushing model to Hub"):
#     if hf_token is not None:  # Optionally save token programmatically
#         HfFolder.save_token(hf_token)
    
#     repo = Repository(local_dir=local_dir, repo_id=repo_id, clone_from=f"https://huggingface.co/{repo_id}")
#     repo.push_to_hub(commit_message=commit_message)

# Example Usage:

# scored_gradients = {}
# for key, score in scores.items():
#     serialized_gradients = get_weights(key)
    
#     total_score = score['loss_score'] + score['perplexity_score']
#     scored_gradients[key] = (gradients, total_score)

# averaged_gradients = average_gradients(scored_gradients)
# apply_averaged_gradients(model, averaged_gradients)

# Define your model's local directory and repository ID
local_dir = args.averager.save_directory #TODO add me to config :)
repo_id = args.averager.hf_repository #TODO add me to config :)

# Save the updated model
model.save_pretrained(local_dir)

# Push the model to the Hugging Face Hub
push_to_hf_hub(local_dir=local_dir, repo_id=repo_id, hf_token=args.averager.hf_token, commit_message=f"Updated model SN25 with {}")#FIXME add block numbers
