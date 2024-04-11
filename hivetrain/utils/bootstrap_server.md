 # DHT Manager Documentation

This Python script is designed to manage a Decentralized Hash Table (DHT) network. It uses the HiveMind library for implementing the DHT functionality and Waitress for running the Flask application as a production-ready web server. This documentation explains the logic behind the provided code.

## Importing Libraries
```python
import argparse
from flask import Flask, jsonify
import hivemind
import time
import random
import threading
import logging
import sys
```
The script starts by importing required libraries and modules:
- `argparse`: For parsing command line arguments.
- `Flask`: A web framework for building web applications in Python.
- `hivemind`: The library used to create and manage DHTs.
- `time`: For handling time-related functionalities.
- `random`: To select a random element from the list.
- `threading`: To create thread-safe locks.
- `logging`: For setting up logging.

## Logging Configuration
```python
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('bootstrap')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```
Logging is configured with a basic logger at the error level. A new handler is created for stdout, and its logging level is set to info. The formatter is defined, and the handler is added to the logger.

## Creating and Managing DHTs
The `check_and_manage_dhts` function checks each DHT in the list for its availability status. If a connection fails, it marks that particular DHT as non-responsive and removes it from the list. The function also creates new DHTs if needed to maintain a count of 10 DHTs in the list.

```python
def check_and_manage_dhts():
    global last_checked

    for dht in dht_list:
        try:
            test_dht = hivemind.DHT(initial_peers=[str(dht.get_visible_maddrs()[0])], start=True)
            test_dht.shutdown()
        except Exception as e:
            dht.terminate()
            dht_list.remove(dht)

    if len(dht_list) < 10:
        initial_peers = [dht.get_visible_maddrs()[0] for dht in dht_list]
        new_dht = hivemind.DHT(host_maddrs=[f"/ip4/{args.host_address}/tcp/0", f"/ip4/{args.host_address}/udp/0/quic"], initial_peers=initial_peers, start=True)
        dht_list.append(new_dht)

    last_checked = time.time()
```

## Flask Application Setup
A new Flask application is created using the `__name__` as the name of the file. An empty list named `dht_list` is used to store interconnected DHTs, and a global variable `last_checked` is initialized with 0. A lock object named `lock` is created for thread-safe access to shared resources.

## Before Request Hook
```python
@app.before_request
def before_request():
    global last_checked

    if (time.time() - last_checked > 100) and len(dht_list) > 0: 
        check_and_manage_dhts()
```
A before request hook is defined, which checks the status of DHTs before handling incoming requests. If more than 10 minutes have passed since the last check or if there are less than ten active DHTs in the list, it calls the `check_and_manage_dhts()` function to update the list and ensure that there are at least 10 active DHTs.

## Routes
The script defines a single route named `/return_dht_address` which returns initial peers addresses for connecting to an available DHT when requested by another node. This route checks if there are any available DHTs in the list, then selects a random one and returns its initial peers as a JSON response. If no available DHTs are found, it creates a new one and adds it to the list before returning its initial peers.

## Main Function
The script runs the Flask application using Waitress instead of the traditional `app.run()` method, which provides better performance, scalability, and production-ready features for serving the web application.

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DHT Manager')
    parser.add_argument('--host_address', type=str, default="0.0.0.0", help='Machine\'s internal IP')
    parser.add_argument('--host_port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--external_address', type=str, default="20.20.20.20", help='Machine\'s external IP')
    args = parser.parse_args()
    serve(app, host=args.host_address, port=args.host_port)
```
The main function initializes an `argparse` object for handling command-line arguments and parses them accordingly. The script then runs the Flask application using Waitress by calling the `serve()` function with the application instance and specified host address and port number as arguments.