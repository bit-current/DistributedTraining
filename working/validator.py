import argparse
from flask import Flask, request, jsonify
import numpy as np
import time
from .auth import authenticate_request
from .config import config

app = Flask(__name__)

model_checksums = {}
metrics_data = {}

last_evaluation_time = time.time()
evaluation_interval = 120

last_evaluation_time = time.time()
evaluation_interval = 120
sync_interval = 600  # Synchronization interval in seconds
last_sync_time = time.time()

# Bittensor wallet and network initialization
wallet = None
subtensor = None
metagraph = None
config = config() #TODO Add our config here

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

    check_registered()

def check_registered():
    global wallet, subtensor, config
    if not subtensor.is_hotkey_registered(netuid=config.netuid, hotkey_ss58=wallet.hotkey.ss58_address):
        print(f"Wallet: {wallet} is not registered on netuid {config.netuid}. Please register the hotkey before trying again")
        exit()

def serve_on_subtensor(external_ip, external_port, netuid, max_retries=5, wait_for_inclusion=True, wait_for_finalization=False):
    global wallet, subtensor
    retry_count = 0
    while retry_count < max_retries:
        try:
            serve_success = subtensor.serve(
                wallet=wallet,
                ip=external_ip,
                port=external_port,
                netuid=netuid,
                protocol=4,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=False,
            )
            if serve_success:
                print(f"Serving on IP: {external_ip}, Port: {external_port}")
                break
            else:
                print("Failed to serve on Subtensor network. Retrying...")
        except Exception as e:
            print(f"Error serving on Subtensor network: {e}")
        
        retry_count += 1
        sleep_time = math.pow(2, retry_count)  # Exponential backoff
        print(f"Retry {retry_count}/{max_retries}. Retrying in {sleep_time} seconds.")
        time.sleep(sleep_time)

    if retry_count == max_retries:
        print("Max retries reached. Failed to serve on Subtensor network.")

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

@app.before_request
def before_request():
    global last_evaluation_time
    # Existing before_request code...
    sync()  # Ensure the validator is synchronized with the network state before processing any request.

def verify_model_checksum(rank, checksum):
    global model_checksums
    model_checksums[rank] = checksum

def detect_metric_anomaly(metric="loss", OUTLIER_THRESHOLD=2):
    global metrics_data
    if not metrics_data:
        return []

    aggregated_metrics = {}
    for rank, data in metrics_data.items():
        for metric_recorded in data:
            if metric in metric_recorded.keys():
                #rank = data['rank']
                if rank in aggregated_metrics:
                    aggregated_metrics[rank].append(metric_recorded[metric])
                else:
                    aggregated_metrics[rank] = [metric_recorded[metric]]

    average_losses = {rank: np.mean(losses) for rank, losses in aggregated_metrics.items()}
    
    losses = np.array(list(average_losses.values()))
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    z_scores = np.abs((losses - mean_loss) / std_loss) if std_loss > 0 else np.zeros_like(losses)
    outliers = np.array([z > OUTLIER_THRESHOLD for z in z_scores])
    
    scores = []
    for i, (rank, _) in enumerate(average_losses.items()):
        score = 0 if outliers[i] else 1
        scores.append({'rank': rank, 'score': score})
    
    for score_data in scores:
        print(f"Rank: {score_data['rank']}, Score: {score_data['score']}")
    
    return scores


def run_evaluation():
    global model_checksums, metrics_data
    checksum_frequencies = {}
    for rank, checksum in model_checksums.items():
        checksum_frequencies[checksum] = checksum_frequencies.get(checksum, 0) + 1
    model_scores = {}
    try:
        most_common_checksum = max(checksum_frequencies, key=checksum_frequencies.get)
        model_scores = {rank: (1 if checksum == most_common_checksum else 0) for rank, checksum in model_checksums.items()}
        print("Model scores based on checksum consensus:", model_scores)

    except ValueError:
        pass

    detect_metric_anomaly()

    model_checksums.clear()
    metrics_data.clear()


@app.before_request
def before_request():
    global last_evaluation_time
    current_time = time.time()
    if current_time - last_evaluation_time > evaluation_interval:
        run_evaluation()
        last_evaluation_time = current_time

@app.route('/validate_model', methods=['POST'])
@authenticate_request
def validate_model():
    data = request.get_json()
    verify_model_checksum(data['rank'], data['checksum'])
    return jsonify({"message": "Model checksum received and verified"})

@app.route('/validate_metrics', methods=['POST'])
@authenticate_request
def validate_metrics():
    data = request.get_json()
    data['rank'] = int(data['rank'])
    if data['rank'] not in metrics_data:
        metrics_data[data['rank']] = []
    metrics_data[data['rank']].append({'rank': data['rank'], 'loss': data['metrics']['loss']})
    return jsonify({"message": "Metrics received"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validator Server')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    initialize_bittensor_objects()
    serve_on_subtensor("127.0.0.1", args.port, netuid, max_retries=5, wait_for_inclusion=True, wait_for_finalization=True) #FIXME hardcoded to localhost

    app.run(host="0.0.0.0", port=args.port)
