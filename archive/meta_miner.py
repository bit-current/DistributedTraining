import argparse
import atexit
import bittensor
import requests
import time
import subprocess
import json
import signal
from hivetrain.config import Configurator
from hivetrain.btt_connector import get_validator_uids_and_addresses, BittensorNetwork, serve_axon


# Existing meta_miner.py code with modifications to use Bittensor wallet for signing and authentication
def create_signed_message(message):
    """Sign a message and return the signature."""
    global wallet
    signature = wallet.hotkey.sign(message).hex()  # Convert bytes to hex string for easy transmission
    public_address = wallet.hotkey.ss58_address
    return message, signature, public_address

def register_with_orchestrator(orchestrator_url):
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

def update_orchestrator(orchestrator_url, miner_id, trigger_error=False):
    """Send an update to the orchestrator, optionally with an error trigger, using PublicKey authentication."""
    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(f"{timestamp}:{miner_id}:{trigger_error}")
    data = {'miner_id': miner_id, 'trigger_error': trigger_error, 'message': message, 'signature': signature, 'public_address': public_address}
    response = requests.post(f"{orchestrator_url}/update", json=data)
    return response.json() if response.ok else None

def get_training_params(orchestrator_url, miner_id):
    """Retrieve training parameters from the orchestrator, using PublicKey authentication."""
    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(f"{timestamp}:{miner_id}")
    data = {'miner_id': miner_id, 'message': message, 'signature': signature, 'public_address': public_address}
    response = requests.post(f"{orchestrator_url}/training_params", json=data)
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
    ]
    if len(validator_urls) > 0:
        cmd += ["--validator-urls"] + validator_urls
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

def sketchy_hack_for_local_runs(validator_ips, known_ip):
    """Replaces known IPs with 127.0.0.1 for local runs."""
    return [ip.replace(known_ip, "127.0.0.1") for ip in validator_ips]

def main(orchestrator_url, miner_script, batch_size, epochs, tcp_store_address, tcp_store_port):
    miner_id = None
    torchrun_process = None

    while True:
        timestamp = str(int(time.time()))
        message, signature, public_address = create_signed_message(timestamp)
        
        if miner_id is not None:
            if torchrun_process and torchrun_process.poll() is not None:
                if torchrun_process.returncode != 0:
                    print(f"Training process ended with error, return code {torchrun_process.returncode}. Notifying orchestrator.")
                    update_response = orchestrator_interface.update_orchestrator(orchestrator_url, {'miner_id': miner_id, 'trigger_error': True})
                    if update_response and update_response.status_code == 200:
                        print("Orchestrator notified about the error.")
                    else:
                        print("Failed to notify orchestrator about the error.")
                    torchrun_process = None
                    time.sleep(10)
                    continue

        # Add check for miner_id existence in Orchestrator response
        if miner_id is None or (update_response and update_response.status_code == 404):
            registration_data = {'message': message, 'signature': signature, 'public_address': public_address}
            registration_response = orchestrator_interface.register_with_orchestrator(orchestrator_url, registration_data)
            if registration_response and registration_response.status_code == 200:
                registration_data = registration_response.json()
                miner_id = registration_data.get('miner_id')
                print(f"Registered with miner ID: {miner_id}")
            else:
                print("Failed to register with orchestrator. Retrying...")
                time.sleep(10)
                continue

        data = {'miner_id': miner_id, 'trigger_error': False, 'message': message, 'signature': signature, 'public_address': public_address}
        update_response = orchestrator_interface.update_orchestrator(orchestrator_url, data)
        if update_response and update_response.status_code == 200:
            update_data = update_response.json()
            state = update_data.get("state")
            print(f"Current state from orchestrator: {state}")

            if state == "training" and not torchrun_process:
                data = {'miner_id': miner_id, 'message': message, 'signature': signature, 'public_address': public_address}
                training_params_response = orchestrator_interface.get_training_params(orchestrator_url, data)
                if training_params_response and training_params_response.status_code == 200:
                    training_params = training_params_response.json()
                    print(f"Starting training with params: {training_params}")
                    _, validator_ips = get_validator_uids_and_addresses(metagraph, config.neuron.vpermit_tao_limit)
                    validator_ips = ["127.0.0.1:3000"]
                    torchrun_process = start_training(training_params['rank'], training_params['world_size'], miner_script, batch_size, epochs, validator_ips, tcp_store_address, tcp_store_port)
                else:
                    print("Failed to get training params. Retrying...")
            elif state == "onboarding" and torchrun_process:
                torchrun_process.terminate()
                torchrun_process.wait()
                torchrun_process = None
                print("Training process terminated, ready for next run.")
        else:
            print("Failed to update orchestrator. Retrying...")
        
        time.sleep(10)

if __name__ == "__main__":
    config = Configurator.combine_configs() #argparse.ArgumentParser(description="Meta Miner Configuration")
    #parser.add_argument("--orchestrator-url", type=str, required=True, help="URL of the orchestrator")
    #parser.add_argument("--miner-script", type=str, default="miner_cpu.py", help="The miner script to execute for training")
    #parser.add_argument("--batch-size", type=int, default=64, help="Batch size per forward/backward pass")
    #parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    #parser.add_argument('--validator-urls', type=str, nargs="+", required=True, help='URLs of the validators for local testing only')

    BittensorNetwork.initialize(config)

    # Now you can access wallet, subtensor, and metagraph like this:
    wallet = BittensorNetwork.wallet
    subtensor = BittensorNetwork.subtensor
    metagraph = BittensorNetwork.metagraph

    #serve_on_subtensor(config.host_address, config.port, config.netuid, max_retries=5, wait_for_inclusion=True, wait_for_finalization=True) #FIXME hardcoded to localhost
    serve_axon(config.netuid,config.host_address,config.host_address, config.port, config.port)

    main(config.orchestrator_url, config.miner_script, config.batch_size, config.epochs, 
        config.tcp_store_address, config.tcp_store_port)