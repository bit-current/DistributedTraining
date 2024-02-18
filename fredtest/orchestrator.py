from flask import Flask, request, jsonify
import threading
import json

app = Flask(__name__)

# This dictionary will store the miner information, including their status and any other necessary details.
miners = {}

@app.route('/register_miner', methods=['POST'])
def register_miner():
    data = request.json
    miner_address = data.get('miner_address')
    if not miner_address:
        return jsonify({"error": "Miner address is required."}), 400
    
    # Here, you'd normally validate the miner's signature to authenticate the request.
    # This example assumes all miners are trustworthy for simplicity.
    
    if miner_address in miners:
        return jsonify({"error": "Miner already registered."}), 400
    else:
        miners[miner_address] = {"status": "active"}
        update_ranks()
        return jsonify({"message": "Miner registered successfully.", "rank": list(miners).index(miner_address), "world_size": len(miners)})

@app.route('/deregister_miner', methods=['POST'])
def deregister_miner():
    data = request.json
    miner_address = data.get('miner_address')
    if not miner_address or miner_address not in miners:
        return jsonify({"error": "Invalid miner address."}), 400
    
    del miners[miner_address]
    update_ranks()
    return jsonify({"message": "Miner deregistered successfully."})

@app.route('/update_world_size', methods=['GET'])
def get_world_size():
    return jsonify({"world_size": len(miners)})

def update_ranks():
    # This function would update the ranks of miners if necessary.
    # For this simple example, ranks are implicitly determined by the order miners register.
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
