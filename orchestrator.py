from flask import Flask, request, jsonify
import threading
import time

app = Flask(__name__)

# Global variable to keep track of the number of connected miners and assign ranks
connected_miners = 0
lock = threading.Lock()

# Endpoint for miners to join the training run
@app.route('/join', methods=['POST'])
def join():
    global connected_miners
    with lock:
        current_rank = connected_miners
        connected_miners += 1
    response = {'rank': current_rank, 'world_size': connected_miners}
    return jsonify(response)

# Endpoint for miners to update their world_size periodically
@app.route('/update', methods=['GET'])
def update():
    return jsonify({'world_size': connected_miners})

# Background thread to decrement the number of connected miners
def decrement_connected_miners():
    global connected_miners
    while True:
        with lock:
            if connected_miners > 0:
                connected_miners -= 1
        time.sleep(10) # Adjust the time interval t as needed

if __name__ == "__main__":
    # Start the background thread
    decrement_thread = threading.Thread(target=decrement_connected_miners)
    decrement_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
