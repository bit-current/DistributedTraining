import argparse
import atexit
import bittensor
import json
import os
import requests
import signal
import subprocess
import time

from hivetrain.config import Configurator
from hivetrain.btt_connector import get_validator_uids_and_addresses, BittensorNetwork, serve_axon

class OrchestratorInterface:
    def __init__(self, network_enabled=True):
        self.network_enabled = network_enabled

    def _send_request(self, url, method='post', data=None, params=None):
        try:
            if self.network_enabled:
                if method == 'post':
                    response = requests.post(url, json=data)
                elif method == 'get':
                    response = requests.get(url, params=params)
                response.raise_for_status()  # Raises a HTTPError for bad responses
                return response
            else:
                return self._mock_response(url, data, params)
        except requests.exceptions.RequestException as e:
            # Handle all request exceptions, including connection errors, timeouts, and HTTP errors
            print(f"Request to {url} failed: {e}")
            return None

    def _mock_response(self, url, data, params):
        mock_responses = {
            '/register': {'status_code': 200, 'json': lambda: {'miner_id': 'mock_miner_id', 'state': 'registered'}},
            '/update': {'status_code': 200, 'json': lambda: {'state': 'updated'}},
            '/training_params': {'status_code': 200, 'json': lambda: {'params': 'mock_params'}}
        }
        response_path = url.split('/')[-1]
        if response_path in mock_responses:
            mock = mock_responses[response_path]
            return type('MockResponse', (object,), {'status_code': mock['status_code'], 'json': mock['json'], 'raise_for_status': lambda: None})
        else:
            return type('MockResponse', (object,), {'status_code': 404, 'json': lambda: {'error': 'Not found'}, 'raise_for_status': lambda: None})

    def register_with_orchestrator(self, orchestrator_url, data):
        return self._send_request(f"{orchestrator_url}/register", method='post', data=data)

    def update_orchestrator(self, orchestrator_url, data):
        return self._send_request(f"{orchestrator_url}/update", method='post', data=data)

    def get_training_params(self, orchestrator_url, data):
        return self._send_request(f"{orchestrator_url}/training_params", method='post', data=data)

# Existing meta_miner.py code with modifications to use Bittensor wallet for signing and authentication
def create_signed_message(message):
    """Sign a message and return the signature."""
    global wallet
    signature = wallet.hotkey.sign(message).hex()  # Convert bytes to hex string for easy transmission
    public_address = wallet.hotkey.ss58_address
    return message, signature, public_address


orchestrator_interface = OrchestratorInterface(network_enabled=True)  # Assuming network is enabled, adjust as necessary


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
        #f"--subtensor.network=train" #FIXME local only pass all args automatically from parent
    ]
    if len(validator_urls) > 0:
        cmd += ["--validator-urls"] + validator_urls

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    return process
    

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
            # Non-blocking check of the subprocess status
            if torchrun_process and torchrun_process.poll() is not None:
                # Process has completed, check for errors
                if torchrun_process.returncode != 0:
                    # Process ended with an error, notify the orchestrator
                    print(f"Training process ended with error, return code {torchrun_process.returncode}. Notifying orchestrator.")
                    update_response = orchestrator_interface.update_orchestrator(orchestrator_url, {'miner_id': miner_id, 'trigger_error': True})
                    if update_response and update_response.status_code == 200:
                        print("Orchestrator notified about the error.")
                    else:
                        print("Failed to notify orchestrator about the error.")
                    torchrun_process = None  # Reset process to allow for restart if needed
                    time.sleep(10)
                    continue 

        if miner_id is None:
            registration_data = {'message': message, 'signature': signature, 'public_address': public_address}
            registration_response = orchestrator_interface.register_with_orchestrator(orchestrator_url, registration_data)
            if registration_response and registration_response.status_code == 200:
                registration_data = registration_response.json()
                miner_id = registration_data.get('miner_id')
                print(f"Registered with miner ID: {miner_id}")
            else:
                print("Failed to register with orchestrator. Retrying...")
                time.sleep(10)  # Wait before retrying registration
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
            miner_id = None
            print("Failed to update orchestrator. Retrying...")
        
        time.sleep(10)  # Polling interval for loop

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
    serve_axon(config.netuid,config.axon.ip,config.axon.external_ip, config.axon.port, config.axon.external_port)

    main(config.meta_miner.orchestrator_url, config.meta_miner.miner_script, config.miner.batch_size, config.miner.epochs, 
        os.environ.get("STORE_ADDRESS", "127.0.0.1"), os.environ.get("STORE_PORT", 4999))