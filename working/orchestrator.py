#from substrateinterface import Keypair
from substrateinterface.utils.ss58 import ss58_decode, ss58_encode 
# or from scalecodec.utils.ss58 import ss58_decode, ss58_encode

from flask import Flask, jsonify, request, abort
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, ed25519
from cryptography.hazmat.primitives import hashes
import threading
import time
import base64
import datetime

app = Flask(__name__)

lock = threading.Lock()
nodes_metadata = {}
current_rank = 0


training_start_time = None  # The time when the current training session starts
training_duration = datetime.timedelta(minutes=30)  # Duration of the training session
wait_duration = datetime.timedelta(minutes=10)  # Duration between training sessions
next_training_time = datetime.datetime.now()  # Time for the next training session to start

def check_training_schedule():
    global next_training_time
    now = datetime.datetime.now()
    if now >= next_training_time - wait_duration:  # Allow joining during wait duration
        # Training session can start soon
        next_training_time = now + training_duration + wait_duration
        return True, 0
    else:
        # Training session cannot start, calculate wait time including wait duration
        wait_time = (next_training_time - wait_duration - now).total_seconds()
        return False, wait_time

@app.route('/check_status', methods=['GET'])
def check_status():
    can_start, wait_time = check_training_schedule()
    if can_start:
        return jsonify({"status": "can_start", "wait_time": 0})
    else:
        return jsonify({"status": "wait", "wait_time": wait_time})

def verify_signature_with_ss58(public_key_id, signature, message):
    try:
        decoded_key = ss58_decode(public_key_id)
        decoded_key_bytes = bytes.fromhex(decoded_key)
        ed25519.Ed25519PublicKey.from_public_bytes(decoded_key_bytes)

    except ValueError as e:
        return False  # Invalid SS58 key
    except Exception as e:
        # Handle other possible exceptions, such as issues with the public key format
        return False
    try:
        # Convert decoded key to the format expected by the cryptography library
        
        
        # Verify the signature
        # Note: This assumes `message` is the original message to be verified against the signature,
        # and `signature` is a byte string of the signature. You might need to adjust the decoding or encoding.
        public_key.verify(
            signature=base64.b64decode(signature),
            data=message.encode(),
            padding=padding.PKCS1v15(),
            algorithm=hashes.SHA256()
        )
        return True  # Signature is valid
    except InvalidSignature:
        return False  # Signature is invalid
    except Exception as e:
        # Handle other possible exceptions, such as issues with the public key format
        return False

def authenticate_request(func):
    def wrapper(*args, **kwargs):
        public_key_id = request.headers.get('Public-Key-Id')
        signature = request.headers.get('Signature')
        if not public_key_id or not signature or not verify_signature_with_ss58(public_key_id, signature, request.data.decode()):
            abort(401, 'Authentication failed')
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

#@authenticate_request
@app.route('/register', methods=['POST'])
def register_node():
    global current_rank
    with lock:
        assigned_rank = current_rank
        current_rank += 1
        nodes_metadata[assigned_rank] = {"registration_time": time.time(), "uptime": 0}
    master_rank = max(nodes_metadata, key=lambda k: nodes_metadata[k].get('uptime', 0))
    return jsonify({"rank": assigned_rank, "world_size": len(nodes_metadata), "master_rank": master_rank})

#@authenticate_request
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

#@authenticate_request
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
