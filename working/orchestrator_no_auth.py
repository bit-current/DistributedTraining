from flask import Flask, jsonify, request
from transitions.extensions import GraphMachine as Machine
import threading
import time

app = Flask(__name__)

lock = threading.Lock()
nodes_metadata = {}
current_meta_rank = 0

class Orchestrator(object):
    pass

states = ['waiting', 'running', 'error_handling', 'updating', 'idle']
transitions = [
    {'trigger': 'start_task', 'source': 'waiting', 'dest': 'running'},
    {'trigger': 'task_completed', 'source': 'running', 'dest': 'updating'},
    {'trigger': 'task_error', 'source': 'running', 'dest': 'error_handling'},
    {'trigger': 'reset_task', 'source': 'error_handling', 'dest': 'running'},
    {'trigger': 'update_checkpoint', 'source': 'updating', 'dest': 'waiting'},
    {'trigger': 'error_unresolved', 'source': 'error_handling', 'dest': 'idle'},
    {'trigger': 'resolve_error', 'source': 'idle', 'dest': 'waiting'}
]

orchestrator = Orchestrator()
machine = Machine(model=orchestrator, states=states, transitions=transitions, initial='waiting', auto_transitions=False)

@app.route('/register', methods=['POST'])
def register_node():
    global current_meta_rank
    with lock:
        assigned_meta_rank = current_meta_rank
        current_meta_rank += 1
        nodes_metadata[assigned_meta_rank] = {"registration_time": time.time(), "uptime": 0}
    master_meta_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0))
    # Trigger state machine to start task if conditions are met
    orchestrator.start_task()
    return jsonify({"meta_rank": assigned_meta_rank, "world_size": len(nodes_metadata), "master_meta_rank": master_meta_rank})

@app.route('/update', methods=['POST'])
def update_node():
    data = request.get_json()
    meta_rank = data['meta_rank']
    uptime = data['uptime']
    with lock:
        if meta_rank in nodes_metadata:
            nodes_metadata[meta_rank]['uptime'] = uptime
        master_meta_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0))
        world_size = len(nodes_metadata)
    # Here you might check conditions to transition to other states
    return jsonify({"world_size": world_size, "master_meta_rank": master_meta_rank})

@app.route('/deregister', methods=['POST'])
def deregister_node():
    data = request.get_json()
    meta_rank = data.get('meta_rank')
    with lock:
        if meta_rank in nodes_metadata:
            del nodes_metadata[meta_rank]
    master_meta_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0)) if nodes_metadata else -1
    # Trigger state machine to handle errors if conditions are met
    orchestrator.task_error()
    return jsonify({"status": "deregistered", "meta_rank": meta_rank, "master_meta_rank": master_meta_rank})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
