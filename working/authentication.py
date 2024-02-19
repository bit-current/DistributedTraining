from flask import Flask, request, jsonify, make_response
from functools import wraps
from substrateinterface import Keypair, KeypairType

def authenticate_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        message = data.get('message') if data else None
        signature = data.get('signature') if data else None
        public_address = data.get('public_address') if data else None

        if not (message and signature and public_address):
            return make_response(jsonify({'error': 'Missing message, signature, or public_address'}), 400)

        if public_address not in public_key_registry:
            return make_response(jsonify({'error': 'Public address not recognized'}), 403)

        keypair_public = Keypair(ss58_address=public_address, crypto_type=KeypairType.SR25519)
        is_valid = keypair_public.verify(message.encode('utf-8'), bytes.fromhex(signature))

        if is_valid:
            return f(*args, **kwargs)
        else:
            return make_response(jsonify({'error': 'Signature verification failed'}), 403)