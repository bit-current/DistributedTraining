from flask import Flask, jsonify, request
import threading
import time

app = Flask(__name__)

lock = threading.Lock()
nodes_metadata = {}
current_rank = 0

@app.route('/register', methods=['POST'])
def register_node():
    global current_rank
    with lock:
        assigned_rank = current_rank
        current_rank += 1
        nodes_metadata[assigned_rank] = {"registration_time": time.time(), "uptime": 0}
    master_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0))
    return jsonify({"rank": assigned_rank, "world_size": len(nodes_metadata), "master_rank": master_rank})

@app.route('/update', methods=['POST'])
def update_node():
    data = request.get_json()
    rank = data['rank']
    uptime = data['uptime']
    with lock:
        if rank in nodes_metadata:
            nodes_metadata[rank]['uptime'] = uptime
        master_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0))
        world_size = len(nodes_metadata)
    return jsonify({"world_size": world_size, "master_rank": master_rank})

@app.route('/deregister', methods=['POST'])
def deregister_node():
    data = request.get_json()
    rank = data.get('rank')
    with lock:
        if rank in nodes_metadata:
            del nodes_metadata[rank]
    master_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0)) if nodes_metadata else -1
    return jsonify({"status": "deregistered", "rank": rank, "master_rank": master_rank})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

