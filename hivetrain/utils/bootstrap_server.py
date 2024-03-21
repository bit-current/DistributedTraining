import argparse
from flask import Flask
import hivemind
import time
import random
import threading

app = Flask(__name__)

# List to store interconnected DHTs
dht_list = []

# Global variable to store the last checked timestamp
last_checked = time.time()

# Create a lock for thread-safe access to shared resources
lock = threading.Lock()

def check_and_manage_dhts():
    global last_checked
    
    with lock:
        # Check the status of each DHT in the list
        for dht in dht_list:
            try:
                # Attempt to connect to the DHT by creating a new disposable DHT
                test_dht = hivemind.DHT(initial_peers=[dht.get_visible_maddrs()[0]], start=True)
                test_dht.shutdown()
            except Exception:
                # If the connection fails, mark the DHT as non-responsive
                dht.terminate()
                dht_list.remove(dht)
        
        # Create new DHTs if needed
        if len(dht_list) < 3:
            initial_peers = [dht.get_visible_maddrs()[0] for dht in dht_list]
            for _ in range(3 - len(dht_list)):
                new_dht = hivemind.DHT(
                    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                    announce_maddrs=[f"/ip4/{args.host_address}/tcp/0", f"/ip4/{args.host_address}/udp/0/quic"],
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
        if time.time() - last_checked > 600:  # 600 seconds = 10 minutes
            check_and_manage_dhts()

@app.route('/return_dht_address')
def return_dht_address():
    # Check if there are any available DHTs
    if dht_list:
        # Choose a random DHT from the list
        random_dht = random.choice(dht_list)
        return random_dht.get_visible_maddrs()[0]
    else:
        # If no DHTs are available, create a new one and return its address
        new_dht = hivemind.DHT(
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            announce_maddrs=[f"/ip4/{args.host_address}/tcp/0", f"/ip4/{args.host_address}/udp/0/quic"],
            start=True
        )
        dht_list.append(new_dht)
        return new_dht.get_visible_maddrs()[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DHT Manager')
    parser.add_argument('--host_address', type=str, default="0.0.0.0", help='Machine\'s external IP')
    parser.add_argument('--host_port', type=int, default=5000, help='Port number (default: 5000)')
    args = parser.parse_args()
    app.run(host=args.host_address, port=args.host_port,threaded=True)