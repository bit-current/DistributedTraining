import argparse
from flask import Flask, request, jsonify
import numpy as np
import time
from hivetrain.auth import authenticate_request_with_bittensor
from hivetrain.config import Configurator
from hivetrain.btt_connector import sync, BittensorNetwork, serve_axon
from hivetrain import __spec_version__
import torch

app = Flask(__name__)

model_checksums = {}
metrics_data = {}

last_evaluation_time = time.time()
evaluation_interval = 120

last_evaluation_time = time.time()
evaluation_interval = 300
sync_interval = 600  # Synchronization interval in seconds
last_sync_time = time.time()
last_update = 0

config = Configurator.combine_configs()

BittensorNetwork.initialize(config)
wallet = BittensorNetwork.wallet
subtensor = BittensorNetwork.subtensor
metagraph = BittensorNetwork.metagraph

def set_weights(scores):
    """
    Sets weights on the blockchain based on the scores from loss averaging.

    Args:
        scores (list): A list of dictionaries containing 'rank' and 'score' for each model.
    """
    try:
        chain_weights = torch.zeros(self.subtensor.subnetwork_n(netuid=self.metagraph.netuid))
        for uid, public_address in enumerate(BittensorNetwork.metagraph.hotkeys):
            
            #rank = score['rank']
            try:
                chain_weights[public_address] = scores[public_address]['score'] #FIXME this is blasphemy 
            except:
                continue

        self.subtensor.set_weights(
            wallet=BittensorNetwork.wallet,
            netuid=BittensorNetwork.metagraph.netuid,
            uids=torch.arange(0, len(chain_weights)),
            weights=chain_weights.to("cpu"),
            wait_for_inclusion=False,
            version_key=__spec_version__,
        )
    except Exception as e:
        print(f"Error setting weights: {e}")

def should_set_weights() -> bool:
    global last_update
    return (BittensorNetwork.subtensor.get_current_block() - last_update) > BittensorNetwork.config.neuron.epoch_length

  # Initialize Validator object properly with required parameters


def verify_model_checksum(public_address, checksum):
    global model_checksums
    model_checksums[public_address] = checksum

def detect_metric_anomaly(metric="loss", OUTLIER_THRESHOLD=2):
    global metrics_data
    if not metrics_data:
        return []

    aggregated_metrics = {}
    for public_address, data in metrics_data.items():
        for metric_recorded in data:
            if metric in metric_recorded.keys():
                #rank = data['rank']
                if public_address in aggregated_metrics:
                    aggregated_metrics[public_address].append(metric_recorded[metric])
                else:
                    aggregated_metrics[public_address] = [metric_recorded[metric]]

    average_losses = {public_address: np.mean(losses) for rank, losses in aggregated_metrics.items()}
    
    losses = np.array(list(average_losses.values()))
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    z_scores = np.abs((losses - mean_loss) / std_loss) if std_loss > 0 else np.zeros_like(losses)
    outliers = np.array([z > OUTLIER_THRESHOLD for z in z_scores])
    
    scores = {}
    for i, (public_address, _) in enumerate(average_losses.items()):
        score = 0 if outliers[public_address] else 1
        scores[public_address]({'public_address': public_address, 'score': score})
    
    for score_data in scores:
        print(f"Public Key: {score_data['public_address']}, Score: {score_data['score']}")
    
    return scores

def run_evaluation():
    global model_checksums, metrics_data

    checksum_frequencies = {}
    for public_address, checksum in model_checksums.items():
        checksum_frequencies[public_address] = checksum_frequencies.get(public_address, 0) + 1
    
    model_scores = {}
    try:
        most_common_checksum = max(checksum_frequencies, key=checksum_frequencies.get)
        model_scores = {public_address: (1 if checksum == most_common_checksum else 0) for public_address, checksum in model_checksums.items()}
        print("Model scores based on checksum consensus:", model_scores)

    except ValueError:
        pass


    scores = detect_metric_anomaly()
    if should_set_weights():
        set_weights(scores)

    model_checksums.clear()
    metrics_data.clear()


@app.before_request
def before_request():
    global last_evaluation_time
    current_time = time.time()
    if current_time - last_evaluation_time > evaluation_interval:
        run_evaluation()
        last_evaluation_time = current_time

    global last_sync_time, sync_interval
    # Existing before_request code...
    last_sync_time = sync(last_sync_time,sync_interval)  # Ensure the validator is synchronized with the network state before processing any request.


@app.route('/validate_model', methods=['POST'])
@authenticate_request_with_bittensor
def validate_model():
    data = request.get_json()
    verify_model_checksum(data['public_address'], data['checksum'])
    return jsonify({"message": "Model checksum received and verified"})

@app.route('/validate_metrics', methods=['POST'])
@authenticate_request_with_bittensor
def validate_metrics():
    data = request.get_json()
    data['rank'] = int(data['rank'])
    if data['rank'] not in metrics_data:
        metrics_data[data['public_address']] = []
    metrics_data[data['public_address']].append({'public_address': data['public_address'], 'loss': data['metrics']['loss']})
    return jsonify({"message": "Metrics received"})

if __name__ == "__main__":
    
    serve_axon(config.netuid,config.axon.ip,config.axon.external_ip, config.axon.port, config.axon.external_port)
    app.run(host="0.0.0.0", port=config.port)
