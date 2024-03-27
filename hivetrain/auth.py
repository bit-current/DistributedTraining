from functools import wraps
from flask import request, make_response, jsonify
import bittensor
from .btt_connector import BittensorNetwork
from . import __spec_version__
from substrateinterface import Keypair, KeypairType
#metagraph = bittensor.metagraph()  # Ensure this metagraph is synced before using it in the decorator.
import logging

logger = logging.getLogger('waitress')
logger.setLevel(logging.DEBUG)


def authenticate_request_with_bittensor(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        message = data.get('message') if data else None
        signature = data.get('signature') if data else None
        public_address = data.get('public_address') if data else None
        miner_version = data.get("miner_version",0)

        if not (message and signature and public_address):
            logger.info(f"Rejected request without auth data")
            return make_response(jsonify({'error': 'Missing message, signature, or public_address'}), 400)
            
        #Check if miner version is correct
        if int(miner_version) < __spec_version__:
            logger.info(f"Rejected request with wrong miner version")
            return make_response(jsonify({'error': f'Miner version is {miner_version} while current minimum version is {str(__spec_version__)}'}), 403)

        # Check if public_address is in the metagraph's list of registered public keys
        if public_address not in BittensorNetwork.metagraph.hotkeys:
            logger.info(f"Miner {public_address} refused. Not registered")
            return make_response(jsonify({'error': 'Public address not recognized or not registered in the metagraph'}), 403)
        # Use Bittensor's wallet for verification
        #wallet = bittensor.wallet(ss58_address=public_address)
        #is_valid = wallet.verify(message.encode('utf-8'), signature, public_address)
        signature_bytes = bytes.fromhex(signature) if isinstance(signature, str) else signature
        keypair_public = Keypair(ss58_address=public_address, crypto_type=KeypairType.SR25519)
        is_valid = keypair_public.verify(message.encode('utf-8'), signature_bytes)
        if is_valid and not BittensorNetwork.rate_limiter(public_address):
            logger.info(f"Rejected request from blacklisted or too frequent address: {public_address}")
            return make_response(jsonify({'error': 'Too many requests or blacklisted'}), 429)
        if is_valid:
            return f(*args, **kwargs)
        else:
            logger.info(f"Miner {public_address} refused. Signature Verification Failed")
            return make_response(jsonify({'error': 'Signature verification failed'}), 403)
    return decorated_function
