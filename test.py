from substrateinterface.utils.ss58 import ss58_decode, ss58_encode 
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ed25519

ss58_address = '5H6Zofhio1R7WJQGdjEUsZByJtiZf5hHxYmH5CRLPW9d13FZ'
decoded_key_hex = ss58_decode(ss58_address)
decoded_key_bytes = bytes.fromhex(decoded_key_hex)  # Convert hex string to bytes

public_key = ed25519.Ed25519PublicKey.from_public_bytes(decoded_key_bytes)
# Now use the bytes to load the public key
breakpoint()