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

parser = argparse.ArgumentParser(description='DHT Manager')
parser.add_argument('--host_address', type=str, default="0.0.0.0", help='Machine\'s internal IP')
parser.add_argument('--host_port', type=int, default=5000, help='Port number (default: 5000)')
parser.add_argument('--external_address', type=str, default="20.20.20.20", help='Machine\'s external IP')
parser.add_argument('--dht_tcp_port', type=int, help='Machine\'s external IP')
parser.add_argument('--dht_udp_port', type=int, help='Machine\'s external IP')
parser.add_argument('--ports_file', type=str, default=None, help='Path to the file containing allowed ports')

args = parser.parse_args()

# List to store interconnected DHTs
dht_list = []

# Global variable to store the last checked timestamp
last_checked = 0#time.time()

# Create a lock for thread-safe access to shared resources
lock = threading.Lock()

def read_allowed_ports(file_path):
    with open(file_path, 'r') as file:
        ports = [int(line.strip()) for line in file]
    return ports


allowed_ports = None
if args.ports_file:
    allowed_ports = read_allowed_ports(args.ports_file)


def get_port_from_addr(addr_str):
    # Example addr_str: /ip4/peer_ip/tcp/peer_dht_port/p2p/12D3KooWE_some_hash...
    parts = addr_str.split('/')
    # Find the index of the protocol (e.g., "tcp") and get the port right after it
    for protocol in ["tcp", "udp"]:
        if protocol in parts:
            proto_idx = parts.index(protocol)
            if proto_idx + 1 < len(parts):
                return int(parts[proto_idx + 1])
    return None

def get_random_ports(allowed_ports, num_ports=1, port_range=(49152, 65535)):
    used_ports = set()
    for dht in dht_list:
        for addr in dht.get_visible_maddrs():
            addr_str = str(addr)
            used_port = get_port_from_addr(addr_str)
            if used_port:
                used_ports.add(used_port)
    
    if allowed_ports is not None:
        available_ports = set(allowed_ports) - used_ports
        if len(available_ports) < num_ports:
            raise ValueError("Not enough available ports to fulfill the request.")
        return random.sample(available_ports, num_ports)
    else:
        # Generate random ports from the given range if allowed_ports is None
        all_possible_ports = set(range(port_range[0], port_range[1] + 1))
        available_ports = list(all_possible_ports - used_ports)
        if len(available_ports) < num_ports:
            raise ValueError("Not enough available ports to fulfill the request.")
        return random.sample(available_ports, num_ports)

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
        logger.info(f"Replacing {5 - len(dht_list)} DHTs")
        for _ in range(5 - len(dht_list)):
            tcp_port, udp_port = get_random_ports(allowed_ports, num_ports=2)
            new_dht = hivemind.DHT(
                host_maddrs=[f"/ip4/{args.host_address}/tcp/{tcp_port}", f"/ip4/{args.host_address}/udp/{udp_port}/quic"],
                use_relay=False,
                announce_maddrs=[f"/ip4/{args.external_address}/tcp/{tcp_port}", f"/ip4/{args.external_address}/udp/{udp_port}"],
                #identity_path=args.local_DHT_file
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
            logger.info(allowed_ports)
            logger.info("Get ports")
            tcp_port, udp_port = get_random_ports(allowed_ports, num_ports=2)
            logger.info("Get DHT")
            new_dht = hivemind.DHT(
                host_maddrs=[f"/ip4/{args.host_address}/tcp/{tcp_port}", f"/ip4/{args.host_address}/udp/{udp_port}/quic"],
                use_relay=False,
                announce_maddrs=[f"/ip4/{args.external_address}/tcp/{tcp_port}", f"/ip4/{args.external_address}/udp/{udp_port}"],
                #announce_maddrs=[f"/ip4/{args.host_address}"],
                start=True
            )
            dht_list.append(new_dht)
            logger.info(new_dht.get_visible_maddrs())
            initial_peers = [str(multiaddr).replace(args.host_address, args.external_address) for multiaddr in new_dht.get_visible_maddrs()]
            return jsonify({"initial_peers":initial_peers})

if __name__ == '__main__':
    
    #app.run(host=args.host_address, port=args.host_port,threaded=True)
    serve(app, host=args.host_address, port=args.host_port)
