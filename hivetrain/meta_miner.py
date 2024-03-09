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

class OrchestratorClient:
    def __init__(self, orchestrator_url = "http://127.0.0.1:5000"):
        self.orchestrator_url = orchestrator_url

    def send_request(self, endpoint, data=None, method='POST'):
        """Generalized method to send requests to the orchestrator."""
        url = f"{self.orchestrator_url}/{endpoint}"
        try:
            if method.upper() == 'POST':
                response = requests.post(url, json=data)
            else:  # Default to GET if not POST
                response = requests.get(url, params=data)
            response.raise_for_status()  # Raise exceptions for HTTP request errors
            return response.json()  # Return the JSON response if successful
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def register_miner(self, registration_data):
        """Register the miner with the orchestrator and return miner_id."""
        response = self.send_request('register', registration_data)
        if response and 'miner_id' in response:
            return response['miner_id']
        else:
            print("Registration failed or miner_id not in response.")
            return None

    def update_orchestrator(self, update_data):
        """Send an update to the orchestrator and return the state."""
        response = self.send_request('update', update_data)
        if response and 'state' in response:
            return response['state']
        else:
            print("Update failed or state not in response.")
            return None

    def get_training_params(self, params_data):
        """Fetch training parameters from the orchestrator."""
        response = self.send_request('training_params', params_data)
        if response:
            return response  # Return the whole response as it includes training params and possibly tcp_store details
        else:
            print("Failed to get training parameters.")
            return None

    def report_training(self, params_data):
        response = self.send_request('report_training', params_data)
        if response.status_code == 200:
            print("Update Succesful.")
        else:
            print("update Failed")

# Existing meta_miner.py code with modifications to use Bittensor wallet for signing and authentication
def create_signed_message(message):
    """Sign a message and return the signature."""
    global wallet
    signature = wallet.hotkey.sign(message).hex()  # Convert bytes to hex string for easy transmission
    public_address = wallet.hotkey.ss58_address
    return message, signature, public_address


orchestrator_interface = OrchestratorClient()  # Assuming network is enabled, adjust as necessary


class TrainingManager:
    def __init__(self, miner_script, batch_size, epochs):
        self.miner_script = miner_script
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_process = None  # Instance attribute to keep track of the current training process
        self.state_of_launch = None

    def start_training_process(self, training_params, tcp_store_address, tcp_store_port,state,
    validator_urls):
        """
        Starts the training process with the provided parameters. If a training process
        is already running, it is terminated before starting a new one.

        Args:
            training_params (dict): A dictionary containing training-specific parameters.
            tcp_store_address (str): The TCP store address provided by the orchestrator.
            tcp_store_port (int): The TCP store port provided by the orchestrator.
        """
        # Check if there's an existing training process and terminate it if necessary
        if self.current_process and self.current_process.poll() is None:
            self.cleanup_on_exit()

        cmd = [
        "python", miner_script,
        f"--rank={str(rank)}",
        f"--epochs={str(epochs)}",
        f"--world-size={str(world_size)}",
        f"--batch-size={str(batch_size)}",
        f"--tcp-store-address={store_address}",
        f"--tcp-store-port={str(store_port)}",
        #f"--subtensor.network=train" #FIXME local only pass all args automatically from parent
        ]
        
        for key, value in training_params.items():
            cmd.append(f"--{key}={value}")

        if len(validator_urls) > 0:
            cmd += ["--validator-urls"] + validator_urls
        # Start a new training process
        self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.state_of_launch = state

    def get_process_output(self):
        """
        Waits for the current training process to complete and captures its stdout.

        Returns:
            str: The stdout captured from the training process, or an empty string if no process is running.
        """
        if self.current_process:
            # Wait for the process to complete and capture stdout
            stdout, _ = self.current_process.communicate()
            return stdout
        else:
            return ""

    def cleanup_on_exit(self):
        """
        Cleans up resources and terminates the current training process if it's still running.
        """
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
            finally:
                self.current_process = None  # Reset the current_process attribute
                self.state_of_launch = None



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

def main_workflow():
    # Load and initialize configuration and network components
    config = ConfigManager.load_configuration()
    ConfigManager.initialize_network_components()

    timestamp = str(int(time.time()))
    message, signature, public_address = create_signed_message(timestamp)

    # Initialize the orchestrator client with the loaded configuration
    orchestrator_client = OrchestratorClient(config.orchestrator_url)

    # Initialize the training manager without specifying tcp_store_address and tcp_store_port
    training_manager = TrainingManager(
        batch_size=config.batch_size, 
        epochs=config.epochs
        # Note: tcp_store_address and tcp_store_port are now omitted
    )

    # Main loop to manage the miner lifecycle
    miner_id = None
    torchrun_process = None
    finished_training = None
    while True:
        if miner_id is None:
            # Register the miner if not already registered
            registration_data = {'message': message, 'signature': signature, 'public_address': public_address}
            miner_id = orchestrator_client.register_miner(registration_data)
            if miner_id:
                print(f"Registered with miner ID: {miner_id}")
            else:
                print("Failed to register with orchestrator. Retrying...")
                time.sleep(10)  # Wait before retrying registration
                continue

        # Update the orchestrator with the current status
        update_data = {
            'miner_id': miner_id, 'trigger_error': False, 'message': message, 'signature': signature, 'public_address': public_address
        }
        state = orchestrator_client.update_orchestrator(update_data)
        if not state:
            print("Failed to update orchestrator. Retrying...")
            miner_id = None  # Reset miner_id to trigger re-registration
            time.sleep(10)
            continue

        # Handle training, filtering, or onboarding states
        if state == "training" or state.startswith("filtering"):
            if state == "training":
                training_manager.miner_script = "training_script.py"
            else:
                training_manager.miner_script = "filtering_script.py"

            if (not torchrun_process and not finished_training) or (training_manager.state_of_launch != state):
                training_params_response = orchestrator_client.get_training_params({
                    # 'params_data' placeholder for actual parameters request data
                })
                finished_training = False
                if training_params_response:
                    world_size = training_params_response.get('world_size')
                    rank = training_params_response.get('rank')
                    tcp_store_address = training_params_response.get('tcp_store_address')
                    tcp_store_port = training_params_response.get('tcp_store_port')
                    # Adjust the training manager to use the new tcp_store_address and tcp_store_port
                    torchrun_process = training_manager.start_training_process(
                        rank, training_params, tcp_store_address, tcp_store_port)
                    report_training_data = json.loads(training_manager.get_process_output())
                    auth_data = {'message': message, 'signature': signature, 'public_address': public_address}
                    report_training_data = {**report_training_data, **auth_data}
                    orchestrator_client.report_training(report_training_data)
                    finished_training = True #FIXME Maybe remove to allow restarts?
                    print(f"{state.capitalize()} process started.")
                else:
                    print(f"Failed to get training parameters for {state} phase. Retrying...")

        elif state == "onboarding" and torchrun_process:
            training_manager.cleanup_on_exit()
            torchrun_process = None
            miner_id = None
            print("Training process terminated, ready for next run.")

        time.sleep(10)  # Polling interval for main loop

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