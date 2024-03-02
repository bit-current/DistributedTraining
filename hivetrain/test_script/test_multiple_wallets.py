def create_wallet(wallet_name):
    wallet = bt.wallet(name=wallet_name)
    wallet.create_new_hotkey(use_password=False)
    wallet.create_new_coldkey(use_password = False)