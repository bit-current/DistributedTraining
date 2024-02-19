from flask import Flask, request, jsonify
import threading
import json

app = Flask(__name__)

# Store registered miners and their info
miners = {}

@app.route('/register_miner', methods=['POST'])
def register_miner():
    data = request.json
    miner_address = data.get('miner_address')
    if miner_address and miner_address not in miners:
        miners[miner_address] = {"status": "active"}
        print(f"Miner {miner_address} registered.")
        return jsonify({"status": "success", "message": "Miner registered", "world_size": len(miners)}), 200
    else:
        return jsonify({"status": "error", "message": "Miner already registered or address missing"}), 400

@app.route('/deregister_miner', methods=['POST'])
def deregister_miner():
    data = request.json
    miner_address = data.get('miner_address')
    if miner_address in miners:
        del miners[miner_address]
        print(f"Miner {miner_address} deregistered.")
        return jsonify({"status": "success", "message": "Miner deregistered"}), 200
    else:
        return jsonify({"status": "error", "message": "Miner not found"}), 404

@app.route('/get_task', methods=['GET'])
def get_task():
    # This endpoint would be called by miners to receive a new task
    # For simplicity, this example does not implement dynamic task generation
    # Assume all miners work on the same task for demonstration
    task = {"indices": list(range(100)), "submit_update": True, "participate_in_aggregation": False}
    return jsonify(task), 200

@app.route('/submit_update', methods=['POST'])
def submit_update():
    # Miners submit their model updates here
    # Placeholder for handling model updates aggregation
    print("Received model update")
    # In a real scenario, implement aggregation logic here
    return jsonify({"status": "success", "message": "Update received"}), 200

@app.route('/partial_aggregate', methods=['POST'])
def partial_aggregate():
    # Placeholder for partial aggregation logic
    print("Received request for partial aggregation")
    # Return a mock aggregated update for demonstration
    aggregated_update = {"weights": "aggregated_weights_placeholder"}
    return jsonify(aggregated_update), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
