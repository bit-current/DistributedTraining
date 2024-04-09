import argparse
from flask import Flask, jsonify
import hivemind
import time
import random
import threading
import logging
import sys

logging.basicConfig(level=logging.ERROR)  # Set default logging level to ERROR for all loggers

logger = logging.getLogger('bootstrap')
logger.setLevel(logging.INFO)

# Create handler for stdout
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

from waitress import serve

app = Flask(__name__)

# List to store interconnected DHTs
dht_list = []

# Global variable to store the last checked timestamp
last_checked = 0#time.time()

# Create a lock for thread-safe access to shared resources
lock = threading.Lock()

def check_and_manage_dhts():
    global last_checked
    
    # Check the status of each DHT in the list
    
    for dht in dht_list:
        try:
            # Attempt to connect to the DHT by creating a new disposable DHT
            logger.info("Is DHT Alive")
            test_dht = hivemind.DHT(initial_peers=[str(dht.get_visible_maddrs()[0])], start=True)
            test_dht.shutdown()
            logger.info("DHT Alive")
        except Exception as e:
            logger.info(f"DHT failed. {e}")
            # If the connection fails, mark the DHT as non-responsive
            dht.terminate()
            dht_list.remove(dht)
    
    # Create new DHTs if needed
    if len(dht_list) < 10:
        initial_peers = [dht.get_visible_maddrs()[0] for dht in dht_list]
        logger.info(f"Replacing {10 - len(dht_list)} DHTs")
        for _ in range(10 - len(dht_list)):
            new_dht = hivemind.DHT(
                host_maddrs=[f"/ip4/{args.host_address}/tcp/0", f"/ip4/{args.host_address}/udp/0/quic"],
                #announce_maddrs=[f"/ip4/{args.host_address}", f"/ip4/{args.host_address}"],
                initial_peers=initial_peers,
                start=True
            )
            new_dht.wait_until_ready()
            dht_list.append(new_dht)
    
    # Update the last checked timestamp
    last_checked = time.time()


@app.before_request
def before_request():
    global last_checked
    
    with lock:
        # Check if more than 10 minutes have passed since the last check
        if (time.time() - last_checked > 100) and len(dht_list) > 0:  # 600 seconds = 10 minutes
            logger.info("Checking DHT Status")
            check_and_manage_dhts()

@app.route('/return_dht_address')
def return_dht_address():
    # Check if there are any available DHTs
    global dht_list
    if dht_list:
        # Choose a random DHT from the list
        logger.debug(f"Request Received. Available DHTs")
        random_dht = random.choice(dht_list)
        initial_peers = [str(multiaddr).replace(args.host_address, args.external_address) for multiaddr in random_dht.get_visible_maddrs()]
        return jsonify({"initial_peers":initial_peers})
    else:
        # If no DHTs are available, create a new one and return its address
        with lock:
            logger.info("Initializing 1st DHT")
            new_dht = hivemind.DHT(
                host_maddrs=[f"/ip4/{args.host_address}/tcp/0", f"/ip4/{args.host_address}/udp/0/quic"],
                #announce_maddrs=[f"/ip4/{args.host_address}"],
                start=True
            )
            dht_list.append(new_dht)
            initial_peers = [str(multiaddr).replace(args.host_address, args.external_address) for multiaddr in new_dht.get_visible_maddrs()]
            return jsonify({"initial_peers":initial_peers})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DHT Manager')
    parser.add_argument('--host_address', type=str, default="0.0.0.0", help='Machine\'s internal IP')
    parser.add_argument('--host_port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--external_address', type=str, default="20.20.20.20", help='Machine\'s external IP')
    args = parser.parse_args()
    #app.run(host=args.host_address, port=args.host_port,threaded=True)
    serve(app, host=args.host_address, port=args.host_port)
