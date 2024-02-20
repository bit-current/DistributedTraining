import argparse
import atexit
import bittensor
import requests
import time
import subprocess
import json
import signal

def initialize_bittensor_objects():
    global wallet, subtensor, metagraph, config
    base_config = copy.deepcopy(config)
    check_config(base_config)

    if base_config.mock:
        wallet = bt.MockWallet(config=base_config)
        subtensor = MockSubtensor(base_config.netuid, wallet=wallet)
        metagraph = MockMetagraph(base_config.netuid, subtensor=subtensor)
    else:
        wallet = bt.wallet(config=base_config)
        subtensor = bt.subtensor(config=base_config)
        metagraph = subtensor.metagraph(base_config.netuid)

    # Ensure the miner's hotkey is registered on the network
    check_registered()

def check_registered():
    if not subtensor.is_hotkey_registered(netuid=config.netuid, hotkey_ss58=wallet.hotkey.ss58_address):
        print(f"Wallet: {wallet} is not registered on netuid {config.netuid}. Please register the hotkey before trying again")
        exit()

def resync_metagraph():
    global metagraph, config, subtensor
    # Fetch the latest state of the metagraph from the Bittensor network
    print("Resynchronizing metagraph...")
    if not config.mock:
        # Update the metagraph with the latest information from the network
        metagraph = subtensor.metagraph(config.netuid)
    else:
        # In a mock environment, simply log the action without actual network interaction
        print("Mock environment detected, skipping actual metagraph resynchronization.")
    print("Metagraph resynchronization complete.")

def should_sync_metagraph():
    global metagraph, last_sync_time
    current_time = time.time()
    return current_time - last_sync_time > sync_interval

def sync():
    global last_sync_time, metagraph
    if should_sync_metagraph():
        # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
        # This method would need to be defined or adapted from the BaseNeuron implementation.
        resync_metagraph()
        last_sync_time = time.time()

def check_uid_availability_in_subnet_with_details(
    metagraph: "bt.metagraph.Metagraph", vpermit_tao_limit: int
) -> List[dict]:
    """Check availability of all UIDs in a given subnet, returning their IP, port numbers, and hotkeys if they are serving and have less than vpermit_tao_limit stake.
    
    Args:
        metagraph (bt.metagraph.Metagraph): Metagraph object.
        vpermit_tao_limit (int): Validator permit tao limit.
    
    Returns:
        List[dict]: List of dicts with details of available UIDs, including their IP, port, and hotkeys.
    """
    available_uid_details = []
    for uid in range(len(metagraph.S)):
        if metagraph.axons[uid].is_serving and metagraph.S[uid] <= vpermit_tao_limit:
            details = {
                "uid": uid,
                "ip": metagraph.axons[uid].ip,
                "port": metagraph.axons[uid].port,#FIXME? axon_ip()?
                "hotkey": metagraph.hotkeys[uid]
            }
            available_uid_details.append(details)
    return available_uid_details

# Existing meta_miner.py code with modifications to use Bittensor wallet for signing and authentication
def create_signed_message(message):
    """Sign a message and return the signature."""
    global wallet
    signature = wallet.sign(message).hex()  # Convert bytes to hex string for easy transmission
    public_address = wallet.ss58_address
    return message, signature, public_address

def register_with_orchestrator(orchestrator_url, keypair):
    """Attempt to register the miner with the orchestrator, using PublicKey authentication."""
    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(timestamp)
    data = {'message': message, 'signature': signature, 'public_address': public_address}
    response = requests.post(f"{orchestrator_url}/register", json=data)
    if response.status_code == 200:
        data = response.json()
        return data.get('miner_id'), data.get('state')
    else:
        return None, response.json().get('error', 'Failed to register')

def update_orchestrator(orchestrator_url, miner_id, keypair, trigger_error=False):
    """Send an update to the orchestrator, optionally with an error trigger, using PublicKey authentication."""
    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(f"{timestamp}:{miner_id}:{trigger_error}")
    data = {'miner_id': miner_id, 'trigger_error': trigger_error, 'message': message, 'signature': signature, 'public_address': public_address}
    response = requests.post(f"{orchestrator_url}/update", json=data)
    return response.json() if response.ok else None

def get_training_params(orchestrator_url, miner_id, keypair):
    """Retrieve training parameters from the orchestrator, using PublicKey authentication."""
    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(f"{timestamp}:{miner_id}")
    params = {'miner_id': miner_id, 'message': message, 'signature': signature, 'public_address': public_address}
    response = requests.get(f"{orchestrator_url}/training_params", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def start_training(rank, world_size, miner_script, batch_size, epochs, validator_urls, store_address, store_port):
    # Ensure all command line arguments are strings
    cmd = [
        "python", miner_script,
        f"--rank={str(rank)}",
        f"--epochs={str(epochs)}",
        f"--world-size={str(world_size)}",
        f"--batch-size={str(batch_size)}",
        f"--store-address={store_address}",
        f"--store-port={str(store_port)}",
    ] + ["--validator-urls"] + validator_urls
    return subprocess.Popen(cmd, shell=False)

def cleanup_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    gone, still_alive = psutil.wait_procs(children, timeout=3, callback=None)
    for alive in still_alive:
        alive.kill()

def main(orchestrator_url, miner_script, batch_size, epochs, validator_urls):
    miner_id = None
    torchrun_process = None

    while True:
        # Always get the current state from the orchestrator
        update_response = update_orchestrator(orchestrator_url, miner_id)
        if update_response:
            state = update_response.get("state")
            miner_id = update_response.get("miner_id", miner_id)  # Update miner_id if provided
            print(f"Current state from orchestrator: {state}")

            if state == "training" and not torchrun_process:
                training_params = get_training_params(orchestrator_url, miner_id)
                if training_params:
                    print(f"Starting training with params: {training_params}")
                    # Extract and use training parameters
                    torchrun_process = start_training(training_params['rank'], training_params['world_size'], miner_script, batch_size, epochs, validator_urls, "127.0.0.1", "4999")
            elif state == "onboarding" and torchrun_process:
                # If the state has reverted to onboarding, terminate any running training process
                torchrun_process.terminate()
                torchrun_process.wait()
                torchrun_process = None
                print("Training process terminated, ready for next run.")
        else:
            # Attempt to register if not already done
            if miner_id is None:
                miner_id, state = register_with_orchestrator(orchestrator_url)
                if miner_id:
                    print(f"Registered with miner ID: {miner_id}, State: {state}")
                else:
                    print("Failed to register with orchestrator.")

        time.sleep(10)  # Polling interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta Miner Configuration")
    parser.add_argument("--orchestrator-url", type=str, required=True, help="URL of the orchestrator")
    parser.add_argument("--miner-script", type=str, default="miner_cpu.py", help="The miner script to execute for training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per forward/backward pass")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--validator-urls', type=str, nargs="+", required=True, help='URLs of the validators')

    args = parser.parse_args()
    main(args.orchestrator_url, args.miner_script, args.batch_size, args.epochs, args.validator_urls)