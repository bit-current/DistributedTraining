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
from hivetrain.miner_manager import MinerManager
from hivetrain.state_manager import StateManager
from hivetrain.store_manager import SubprocessHandler

app = Flask(__name__)

# Configure logging for the orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# wallet = btt.wallet
# subtensor = btt.subtensor
# metagraph = btt.metagraph

class Orchestrator:
    def __init__(self, training_threshold=3):
        self.lock = threading.Lock()
        self.max_inactive_time = 45  # seconds
        self.miner_manager = MinerManager()
        self.onboarding_time_limit = 300  # seconds for onboarding timeout
        self.onboarding_start_time = time.time()
        self.subprocess_handler = SubprocessHandler()
        self.store_address = os.environ.get("STORE_ADDRESS", "127.0.0.1")
        self.store_port = int(os.environ.get("STORE_PORT", 4999))

    def register_or_update_miner(self, public_address, trigger_error=False):
        with self.lock:
            current_time = time.time()
            
            self.cleanup_inactive_miners(current_time)

            if public_address not in self.public_address_to_miner_id:
                miner_id = self.register_new_miner(public_address, current_time)
            else:
                miner_id = self.update_existing_miner(public_address, current_time, trigger_error)

            self.log_activity(current_time)
            self.check_for_state_transition(current_time)

            return miner_id

    def register_new_miner(self, public_address, current_time):
        miner_id = self.rank_counter
        self.rank_counter += 1
        self.public_address_to_miner_id[public_address] = miner_id
        self.meta_miners[miner_id] = {
            "public_address": public_address,
            "first_seen": current_time,
            "last_seen": current_time,
            "total_uptime": 0
        }
        self.onboarding_start_time = current_time
        return miner_id

    def update_existing_miner(self, public_address, current_time, trigger_error):
        miner_id = self.public_address_to_miner_id[public_address]
        if miner_id in self.meta_miners:
            self._update_miner_info(miner_id, current_time)
            if trigger_error and StateManager.state == StateManager.TRAINING:
                self._trigger_error_state()
            return miner_id
        else:
            return None

    def _update_miner_info(self, miner_id, current_time):
        elapsed_time = current_time - self.meta_miners[miner_id]["last_seen"]
        self.meta_miners[miner_id]["last_seen"] = current_time
        self.meta_miners[miner_id]["total_uptime"] += elapsed_time

    def cleanup_inactive_miners(self, current_time):
        with self.lock:
            inactive_miners = [miner_id for miner_id, miner_data in self.meta_miners.items() if current_time - miner_data["last_seen"] > self.max_inactive_time]
            for miner_id in inactive_miners:
                public_address = self.meta_miners[miner_id]["public_address"]
                del self.meta_miners[miner_id]
                del self.public_address_to_miner_id[public_address]
            if len(inactive_miners) != 0 and StateManager.state == StateManager.TRAINING:
                StateManager.transition_to_onboarding()
                

    def log_activity(self, current_time):
        world_size = len(self.meta_miners)
        countdown = self.onboarding_time_limit - (current_time - StateManager.onboarding_start_time) if StateManager.state == StateManager.onboarding else "N/A"
        logging.info(f"Current world size: {world_size}, Countdown to training: {countdown}, State: {StateManager.state}, Target world size: {StateManager.training_state_threshold}")

    def check_for_state_transition(self, current_time):
        if StateManager.state == StateManager and (len(self.meta_miners) >= StateManager.training_state_threshold and current_time - self.onboarding_start_time >= self.onboarding_time_limit):
            self._transition_to_training(current_time)

    def _trigger_error_state(self):
        StateManager.transition_to_onboarding()
        

    def _transition_to_training(self, current_time):
        if self.tcp_store_subprocess:
            self.tcp_store_subprocess.terminate()
            self.tcp_store_subprocess.wait()
        self.subprocess_handler.start_tcp_store(self.store_address, self.store_port)
        StateManager.transition_to_training()
        

    def cleanup(self):
        self.subprocess_handler.cleanup()

orchestrator = Orchestrator()

@app.route('/register', methods=['POST'])
@authenticate_request_with_bittensor
def register_miner():
    if orchestrator.state == "training":
        return jsonify({"error": "Registration closed during training"}), 403
    public_address = request.json.get('public_address')
    new_miner_id = orchestrator.register_or_update_miner(public_address)
    if new_miner_id is not None:
        return jsonify({"miner_id": new_miner_id, "state": orchestrator.state}), 200
    else:
        return jsonify({"error": "Failed to register or update miner"}), 404

@app.route('/update', methods=['POST'])
@authenticate_request_with_bittensor
def update():
    public_address = request.json.get('public_address')
    trigger_error = request.json.get("trigger_error", False)
    updated_miner_id = orchestrator.register_or_update_miner(public_address, trigger_error)
    if updated_miner_id is not None:
        return jsonify({"message": "Miner updated", "miner_id": updated_miner_id, "state": orchestrator.state}), 200
    return jsonify({"error": "Update not processed"}), 400

@app.route('/training_params', methods=['POST'])
@authenticate_request_with_bittensor
def training_params():
    if orchestrator.state != "training":
        return jsonify({"error": "Not in training state"}), 400
    public_address = request.json.get('public_address')
    miner_id = orchestrator.public_address_to_miner_id.get(public_address)
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
    try:
        app.run(debug=True, host='0.0.0.0', port=config.port)
    finally:
        orchestrator.cleanup()

