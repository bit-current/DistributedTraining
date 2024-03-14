from functools import wraps
from flask import request, make_response, jsonify
import bittensor
from hivetrain.btt_connector import BittensorNetwork
from substrateinterface import Keypair, KeypairType
#metagraph = bittensor.metagraph()  # Ensure this metagraph is synced before using it in the decorator.


def authenticate_request_with_bittensor(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        message = data.get('message') if data else None
        signature = data.get('signature') if data else None
        public_address = data.get('public_address') if data else None

        if not (message and signature and public_address):
            return make_response(jsonify({'error': 'Missing message, signature, or public_address'}), 400)

        # Check if public_address is in the metagraph's list of registered public keys
        if public_address not in BittensorNetwork.metagraph.hotkeys:
            return make_response(jsonify({'error': 'Public address not recognized or not registered in the metagraph'}), 403)

        # Use Bittensor's wallet for verification
        #wallet = bittensor.wallet(ss58_address=public_address)
        #is_valid = wallet.verify(message.encode('utf-8'), signature, public_address)
        signature_bytes = bytes.fromhex(signature) if isinstance(signature, str) else signature
        keypair_public = Keypair(ss58_address=public_address, crypto_type=KeypairType.SR25519)
        is_valid = keypair_public.verify(message.encode('utf-8'), signature_bytes)
        if is_valid:
            return f(*args, **kwargs)
        else:
            return make_response(jsonify({'error': 'Signature verification failed'}), 403)
    
    return decorated_function
