import os
from substrateinterface import Keypair
from typing import List

def generate_multiple_wallets(n: int) -> List[dict]:
    """
    Generates multiple wallets, each with a hot and cold keypair, without encryption or user prompts.

    Args:
    - n (int): The number of wallets to generate.

    Returns:
    - List[dict]: A list of dictionaries, each containing the mnemonic, hotkey, and coldkey for each wallet.
    """
    wallets = []

    for _ in range(n):
        # Generate mnemonics for hot and cold keys
        cold_mnemonic = Keypair.generate_mnemonic()
        hot_mnemonic = Keypair.generate_mnemonic()

        # Create keypairs from mnemonics
        cold_keypair = Keypair.create_from_mnemonic(cold_mnemonic)
        hot_keypair = Keypair.create_from_mnemonic(hot_mnemonic)

        # Store wallet info
        wallet_info = {
            "cold_mnemonic": cold_mnemonic,
            "cold_keypair": {
                "ss58_address": cold_keypair.ss58_address,
                "public_key": cold_keypair.public_key.hex(),
            },
            "hot_mnemonic": hot_mnemonic,
            "hot_keypair": {
                "ss58_address": hot_keypair.ss58_address,
                "public_key": hot_keypair.public_key.hex(),
            },
        }
        wallets.append(wallet_info)

    return wallets

if __name__ == '__main__':
    # Example: Generate 3 wallets
    wallets = generate_multiple_wallets(3)
    for wallet in wallets:
        print(wallet)
