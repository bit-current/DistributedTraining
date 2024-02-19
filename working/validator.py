import argparse
from flask import Flask, request, jsonify
import numpy as np
import time

app = Flask(__name__)

model_checksums = {}
metrics_data = {}

last_evaluation_time = time.time()
evaluation_interval = 120

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
def validate_model():
    data = request.get_json()
    verify_model_checksum(data['rank'], data['checksum'])
    return jsonify({"message": "Model checksum received and verified"})

@app.route('/validate_metrics', methods=['POST'])
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

    app.run(host="0.0.0.0", port=args.port)
