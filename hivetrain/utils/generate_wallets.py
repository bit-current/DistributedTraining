import os
from substrateinterface import Keypair
from typing import List
import bittensor as bt
from tqdm import tqdm
from hivetrain.config import Configurator
import time

def generate_multiple_wallets(n: int, main_wallet_mnemonic: str, subtensor: bt.subtensor, reg_amount: int = 0.0001,
    netuid: int = 100) -> List[dict]:
    """
    Generates multiple wallets, each with a hot and cold keypair, without encryption or user prompts.

    Args:
    - n (int): The number of wallets to generate.

    Returns:
    - List[dict]: A list of dictionaries, each containing the mnemonic, hotkey, and coldkey for each wallet.
    """
    wallets = []
    core_tao_wallet = bt.wallet(name="faucet_source", hotkey="faucet_hot", path=".")
    core_tao_wallet.regen_coldkey(use_password=False, overwrite=True, mnemonic=main_wallet_mnemonic)
    for wallet_number in tqdm(range(n)):
        # Generate mnemonics for hot and cold keys
        wallet_of_tao = bt.wallet(name=f"test_coldkey_{wallet_number}", hotkey=f"test_hotkey_{wallet_number}", path=".")
        wallet_of_tao.new_coldkey(use_password=False, overwrite=True)
        wallet_of_tao.new_hotkey(overwrite=True)
        bt.extrinsics.transfer.transfer_extrinsic(subtensor, core_tao_wallet, wallet_of_tao.coldkey.ss58_hotkey, reg_amount, 
            wait_for_inclusion=True, 
            wait_for_finalization=True, 
            keep_alive=True, 
            prompt=False)
        time.sleep(0.5) #Make sure subnet hyperparams allow lots of regs
        bt.extrinsics.registration.burned_register_extrinsic(subtensor, 
        wallet_of_tao, 
        netuid, 
        wait_for_inclusion=True, 
        wait_for_finalization=True, 
        prompt=False)
        

    return wallets

if __name__ == '__main__':
    # Example: Generate 3 wallets
    config = Configurator.combine_configs()
    MAIN_WALLET_MNEMONIC = "ENTER THE MNEMONIC HERE"
    wallets = generate_multiple_wallets(3, MAIN_WALLET_MNEMONIC, config.subtensor, 0.00000001)


        
        