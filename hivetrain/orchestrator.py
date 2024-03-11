import logging
import os
import signal
import subprocess
import threading
import time

from flask import Flask, jsonify, request

from hivetrain.auth import authenticate_request_with_bittensor
from hivetrain.btt_connector import BittensorNetwork, initialize_bittensor_objects
from hivetrain.config import Configurator
from hivetrain.orchestrator_core import Orchestrator
from hivetrain.state_manager import StateManager
from hivetrain.store_manager import SubprocessHandler

app = Flask(__name__)

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

orchestrator = Orchestrator()


@app.route("/register", methods=["POST"])
@authenticate_request_with_bittensor
def register_miner():
    if orchestrator.state == "training" or orchestrator.state == "filtering":
        return (
            jsonify({"error": "Registration closed during training and filtering"}),
            403,
        )
    public_address = request.json.get("public_address")
    new_miner_id = orchestrator.register_or_update_miner(public_address)
    if new_miner_id is not None:
        return jsonify({"miner_id": new_miner_id, "state": orchestrator.state}), 200
    else:
        return jsonify({"error": "Failed to register or update miner"}), 404


@app.route("/update", methods=["POST"])
@authenticate_request_with_bittensor
def update():
    public_address = request.json.get("public_address")
    updated_miner_id = orchestrator.register_or_update_miner(public_address)
    if (
        updated_miner_id is not None
        and updated_miner_id not in orchestrator.blacklisted_miners
    ):
        return (
            jsonify(
                {
                    "message": "Miner updated",
                    "miner_id": updated_miner_id,
                    "state": orchestrator.state,
                }
            ),
            200,
        )
    return jsonify({"error": "Update not processed"}), 400


@app.route("/trigger_error", methods=["POST"])
@authenticate_request_with_bittensor
def trigger_error():
    public_address = request.json.get("public_address")
    if orchestrator.trigger_error_for_miner(public_address):
        return (
            jsonify(
                {"message": "Error triggered successfully", "state": orchestrator.state}
            ),
            200,
        )
    else:
        return jsonify({"error": "Failed to trigger error for miner"}), 400


@app.route("/training_params", methods=["POST"])
@authenticate_request_with_bittensor
def training_params():
    if orchestrator.state == "training":
        miner_id = orchestrator.public_address_to_miner_id.get(public_address)
        if miner_id in orchestrator.meta_miners:
            return (
                jsonify(
                    {
                        "world_size": len(orchestrator.meta_miners),
                        "rank": miner_id,
                        "state": orchestrator.state,
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "Miner not found"}), 404
    elif orchestrator.state.startswith("filtering"):
        pass
        # IF the miner is in the senders list assign a rank of 0
        # Otherwise assign a rank of 1
    else:
        return jsonify({"error": "Not in training state"}), 400
    public_address = request.json.get("public_address")


@app.route("/training_report", methods=["POST"])
@authenticate_request_with_bittensor
def training_params():
    if orchestrator.state == "training":
        metrics = request.json.get("metrics")
        public_address = request.get("public_address")
    elif orchestrator.state.startswith("filtering"):
        # If miners is in the receivers list evaluate
        # otherwise ignore
        metrics = request.json.get("hash")
        public_address = request.get("public_address")
    else:
        return jsonify({"error": "Not in training/filtering state"}), 400


if __name__ == "__main__":
    config = Configurator.combine_configs()

    BittensorNetwork.initialize(config)

    # Now you can access wallet, subtensor, and metagraph like this:
    wallet = BittensorNetwork.wallet
    subtensor = BittensorNetwork.subtensor
    metagraph = BittensorNetwork.metagraph
    try:
        app.run(debug=True, host="0.0.0.0", port=config.port)
    finally:
        orchestrator.cleanup()
