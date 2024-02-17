# validator.py
import argparse
from flask import Flask, request
import numpy as np
import time

app = Flask(__name__)

# Define variables to track time
last_evaluation_time = time.time()
evaluation_interval = 5  # Evaluate every 60 seconds

# Store metrics data
metrics_data = []

def clear_or_log_data():
 # Clear or log the stored metrics data here
    print("Clearing or logging data...")
    print(metrics_data)
    # Clear the metrics data for the next timestep
    metrics_data.clear()

def detect_anomaly():
    losses = metrics_data
    OUTLIER_THRESHOLD = 2 #FIXME

    # Calculate mean and standard deviation of the losses for responders.
    if losses:  # Ensure there are any losses to calculate stats on.
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        # Calculate z-scores based on these stats.
        z_scores = np.abs((losses - mean_loss) / std_loss) if std_loss > 0 else np.zeros_like(losses)
    else:
        mean_loss, std_loss, z_scores = 0, 0, np.array([])

    # Initialize scores with a default value (e.g., 1) for all.
    scores = np.ones(len(losses))
    # Apply a penalty based on the z-score for outliers among responders.
    outliers = np.array([z > OUTLIER_THRESHOLD for z in z_scores])
    scores = np.where(outliers, 0.3, 1)  # Update scores for responders based on outlier status.

    # Assign a score of 0 to non-responders.
    
    print(f"Scores of losses anomaly detection {scores}")

def verify_model_checksum():
    global metrics_data
    # Check if more than 70% of the model checksums are in consensus
    consensus_threshold = len(metrics_data) * 0.7
    unique_checksums = set(metrics_data)
    if len(unique_checksums) >= consensus_threshold:
        print("Model checksum consensus reached.")
    else:
        print("Model checksum consensus not reached. Penalizing non-conforming models.")
        # Penalize any models that do not conform
        non_conforming_models = [checksum for checksum in unique_checksums if metrics_data.count(checksum) < consensus_threshold]
        print(f"Non-conforming models: {non_conforming_models}")


def verify_model_checksum():
    # Perform model checksum consensus here
    pass

def run_evaluation():
    # Define your evaluation function
    print("Running evaluation...")
    # Example: Loss anomaly detection
    detect_anomaly()
    # Example: Model checksum consensus
    verify_model_checksum()
    # Clear or log the data for the next timestep
    clear_or_log_data()

# Use the before_request decorator to check if it's time to run the evaluation function
@app.before_request
def check_evaluation():
    global last_evaluation_time

    # Get the current time
    current_time = time.time()

    # Check if the evaluation interval has passed since the last evaluation
    if current_time - last_evaluation_time >= evaluation_interval:
        # Run the evaluation function
        run_evaluation()

        # Update the last evaluation time
        last_evaluation_time = current_time


@app.route('/validate_gradient', methods=['POST'])
def validate_gradient():
    data = request.get_json()
    #print(f"Received gradient checksum: {data['checksum']}")
    return "OK"

@app.route('/validate_model', methods=['POST'])
def verify_model():
    data = request.get_json()
    #print(f"Received model checksum from rank {data['rank']}: {data['checksum']}")
    return "OK"

@app.route('/validate_metrics', methods=['POST'])
def verify_metrics():
    data = request.get_json()
    for metric, value in data.items():
        pass
        #print(f"Received metric {metric} with values {value} from rank {data['rank']}: {data['checksum']}")
    return "OK"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FSDP Validator')
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
