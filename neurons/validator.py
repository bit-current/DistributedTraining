import threading
import bittensor as bt
import argparse
from flask import Flask, request, jsonify
import numpy as np
import time
from hivetrain.auth import authenticate_request_with_bittensor
from hivetrain.config import Configurator
from hivetrain.btt_connector import sync, BittensorNetwork, serve_axon
from hivetrain import __spec_version__
import torch
import logging

logger = logging.getLogger('waitress')
logger.setLevel(logging.DEBUG)

from waitress import serve


app = Flask(__name__)

#model_checksums = {}
#metrics_data = {}

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

model_checksums_lock = threading.Lock()
metrics_data_lock = threading.Lock()
evaluation_time_lock = threading.Lock()
sync_time_lock = threading.Lock()


# def verify_model_checksum(public_address, checksum):
#     global model_checksums
#     with model_checksums_lock:
#         model_checksums[public_address] = checksum

# def detect_metric_anomaly(metric="loss", OUTLIER_THRESHOLD=2):
#     global metrics_data
#         # Check if metrics_data is empty
#     if not metrics_data:
#         return {}

#     # Initialize a dictionary to aggregate metrics by public_address
#     aggregated_metrics = {}

#     # Aggregate metric values by public_address
#     for public_address, data in metrics_data.items():
#         if metric in data:
#             if public_address in aggregated_metrics:
#                 aggregated_metrics[public_address].append(data[metric])
#             else:
#                 aggregated_metrics[public_address] = [data[metric]]

#     # Calculate average and standard deviation of the metric for each public_address
#     average_metrics = {
#     addr: np.nanmean([val for val in vals if isinstance(val, (int, float, np.float32, np.float64))])
#     for addr, vals in aggregated_metrics.items()
#     }
#     losses = np.array(list(average_metrics.values()))
#     mean_loss = np.mean(losses)
#     std_loss = np.std(losses)

#     # Determine outliers based on the OUTLIER_THRESHOLD
#     is_outlier = {}
#     for addr, avg_loss in average_metrics.items():
#         z_score = abs((avg_loss - mean_loss) / std_loss) if std_loss > 0 else 0
#         is_outlier[addr] = z_score > OUTLIER_THRESHOLD

#     scores = {}
#     for i, (public_address, _) in enumerate(average_metrics.items()):
#         score = 0 if is_outlier[public_address] else 1
#         scores[public_address]({'public_address': public_address, 'score': score})
    
#     for score_data in scores:
#         print(f"Public Key: {score_data['public_address']}, Score: {score_data['score']}")
    
#     return scores




@app.before_request
def before_request():

    with evaluation_time_lock:
        global last_evaluation_time, config
        current_time = time.time()
        if current_time - last_evaluation_time > evaluation_interval:
            BittensorNetwork.run_evaluation()
            last_evaluation_time = current_time

    global last_sync_time, sync_interval
    # Existing before_request code...
    with sync_time_lock:
        last_sync_time = sync(last_sync_time,sync_interval, config)  # Ensure the validator is synchronized with the network state before processing any request.


@app.route('/validate_model', methods=['POST'])
@authenticate_request_with_bittensor
def validate_model():
    data = request.get_json()
    verify_model_checksum(data['public_address'], data['checksum'])
    logging.info(f"Received model validation data from {data['public_address']}")
    return jsonify({"message": "Model checksum received and verified"})

@app.route('/validate_metrics', methods=['POST'])
@authenticate_request_with_bittensor
def validate_metrics():
    data = request.get_json()
    #data['rank'] = int(data['rank'])
    with metrics_data_lock:
        if data['public_address'] not in BittensorNetwork.metrics_data:
            BittensorNetwork.metrics_data[data['public_address']] = {"loss":[]}
        try:
            BittensorNetwork.metrics_data[data['public_address']]['loss'].append(data['metrics']['loss'])
        except Exception as e:
            logger.warning(f"Failed to add data from {data['public_address']} due to {e}")
            #BittensorNetwork.metrics_data[data['public_address']]['loss'] = [data['metrics']['loss']]
    logger.info(f"Received model metrics data from {data['public_address']}")
    return jsonify({"message": "Metrics received"})

if __name__ == "__main__":
    #breakpoint()
    axon = bt.axon(wallet=wallet, config=config)
    axon.serve(netuid=config.netuid, subtensor=subtensor)    
    serve(app, host=config.flask.host_address, port=config.flask.host_port)
