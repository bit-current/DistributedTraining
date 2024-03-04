from flask import Flask, jsonify, request
import threading
import time
import subprocess
import os
import signal
import logging
from hivetrain.auth import authenticate_request_with_bittensor
from hivetrain.config import Configurator
from hivetrain.btt_connector import initialize_bittensor_objects, BittensorNetwork

app = Flask(__name__)

# Configure logging for the orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# wallet = btt.wallet
# subtensor = btt.subtensor
# metagraph = btt.metagraph



def setupTCPStore(store_address, store_port, timeout = 30):
    try:
        # Define the command to launch the TCPStore server script
        command = ['python', 'tcp_store_server.py', store_address, str(store_port), str(timeout)]
        # Launch the TCPStore server as a subprocess
        process = subprocess.Popen(command, shell=False)
        return process
    except Exception as e:
        logging.error(f"Failed to launch TCPStore subprocess: {e}")
        return None

def check_subprocess(process):
    if process.poll() is not None:  # Check if the subprocess has terminated
        # Read output and error streams
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            logging.info(f"TCPStore subprocess exited successfully: {stdout.decode()}")
        else:
            logging.error(f"TCPStore subprocess exited with errors: {stderr.decode()}")

class Orchestrator:
    def __init__(self, training_threshold=3):
        self.lock = threading.Lock()
        self.meta_miners = {}
        self.state = "onboarding"
        self.rank_counter = 0
        self.training_state_threshold = training_threshold
        self.max_inactive_time = 45  # seconds
        self.onboarding_time_limit = 300  # seconds for onboarding timeout
        self.onboarding_start_time = time.time()
        self.tcp_store_subprocess = None
        self.store_address = os.environ.get("STORE_ADDRESS", "127.0.0.1")
        self.store_port = int(os.environ.get("STORE_PORT", 4999))

    def register_or_update_miner(self, miner_id=None, trigger_error=False):
        with self.lock:
            current_time = time.time()
            
            if self.block_new_registrations(miner_id):
                return None

            if miner_id is not None:
                updated_miner_id = self.update_existing_miner(miner_id, current_time, trigger_error)
            else:
                updated_miner_id = self.register_new_miner(current_time)

            self.log_activity(current_time)
            self.check_for_state_transition(current_time)

            return updated_miner_id

    def block_new_registrations(self, miner_id):
        return miner_id is None and self.state == "training"

    def update_existing_miner(self, miner_id, current_time, trigger_error):
        if miner_id in self.meta_miners:
            self._update_miner_info(miner_id, current_time)
            if trigger_error and self.state == "training":
                self._trigger_error_state()
            return miner_id
        else:
            return None

    def register_new_miner(self, current_time):
        miner_id = self.rank_counter
        self.rank_counter += 1
        self.meta_miners[miner_id] = {
            "first_seen": current_time,
            "last_seen": current_time,
            "total_uptime": 0
        }
        self.onboarding_start_time = current_time
        return miner_id

    def log_activity(self, current_time):
        world_size = len(self.meta_miners)
        countdown = self.onboarding_time_limit - (current_time - self.onboarding_start_time) if self.state == "onboarding" else "N/A"
        logging.info(f"Current world size: {world_size}, Countdown to training: {countdown}, State: {self.state}, Target world size: {self.training_state_threshold}")

    def check_for_state_transition(self, current_time):
        if self.state == "onboarding" and (len(self.meta_miners) >= self.training_state_threshold and current_time - self.onboarding_start_time >= self.onboarding_time_limit):
            self._transition_to_training(current_time)

    def _update_miner_info(self, miner_id, current_time):
        elapsed_time = current_time - self.meta_miners[miner_id]["last_seen"]
        self.meta_miners[miner_id]["last_seen"] = current_time
        self.meta_miners[miner_id]["total_uptime"] += elapsed_time

    def _trigger_error_state(self):
        self.state = "onboarding"
        self.onboarding_start_time = time.time()

    def _transition_to_training(self, current_time):
        if self.tcp_store_subprocess:
            self.tcp_store_subprocess.terminate()
            self.tcp_store_subprocess.wait()
        self.tcp_store_subprocess = setupTCPStore(self.store_address, self.store_port)
        self.state = "training"
        self.onboarding_start_time = current_time  # Optionally reset onboarding start time here
        

def cleanup_inactive_miners(self):
    with self.lock:
        current_time = time.time()
        inactive_miners = [miner_id for miner_id, miner_data in self.meta_miners.items() if current_time - miner_data["last_seen"] > self.max_inactive_time]
        for miner_id in inactive_miners:
            del self.meta_miners[miner_id]
        # Reset to onboarding if no active miners 
        if len(inactive_miners) !=0 and self.state == "training":
            self.state = "onboarding"
            self.onboarding_start_time = time.time()

def cleanup():
    # Method to clean up the subprocess when the orchestrator is terminated or when needed
    if self.tcp_store_subprocess:
        self.tcp_store_subprocess.terminate()
        try:
            self.tcp_store_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if not terminated after timeout
            os.kill(self.tcp_store_subprocess.pid, signal.SIGKILL)

orchestrator = Orchestrator()

@app.route('/register', methods=['POST'])
@authenticate_request_with_bittensor
def register_miner():
    if orchestrator.state == "training":
        return jsonify({"error": "Registration closed during training"}), 403
    data = request.json
    miner_id = data.get('miner_id', None)
    new_miner_id = orchestrator.register_or_update_miner(miner_id)
    if new_miner_id is not None:
        return jsonify({"miner_id": new_miner_id, "state": orchestrator.state}), 200
    else:
        return jsonify({"error": "Failed to register or update miner"}), 404

@app.route('/update', methods=['POST'])
@authenticate_request_with_bittensor
def update():
    data = request.json
    miner_id = data.get('miner_id')
    trigger_error = data.get("trigger_error", False)
    updated_miner_id = orchestrator.register_or_update_miner(miner_id, trigger_error)
    if updated_miner_id is not None:
        return jsonify({"message": "Miner updated", "miner_id": updated_miner_id, "state": orchestrator.state}), 200
    return jsonify({"error": "Update not processed"}), 400

@app.route('/training_params', methods=['POST'])
@authenticate_request_with_bittensor
def training_params():
    if orchestrator.state != "training":
        return jsonify({"error": "Not in training state"}), 400
    data = request.json
    miner_id = int(data.get('miner_id'))
    if miner_id in orchestrator.meta_miners:
        return jsonify({"world_size": len(orchestrator.meta_miners), "rank": miner_id, "state": orchestrator.state}), 200
    else:
        return jsonify({"error": "Miner not found"}), 404

if __name__ == "__main__":
    config = Configurator.combine_configs()

    BittensorNetwork.initialize(config)

    # Now you can access wallet, subtensor, and metagraph like this:
    wallet = BittensorNetwork.wallet
    subtensor = BittensorNetwork.subtensor
    metagraph = BittensorNetwork.metagraph

    app.run(debug=True, host='0.0.0.0', port=config.port)

