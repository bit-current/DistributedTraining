class MinerManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.meta_miners = {}
        self.public_address_to_miner_id = {}
        self.rank_counter = 0
        self.max_inactive_time = 20

    