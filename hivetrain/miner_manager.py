class MinerManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.meta_miners = {}
        self.public_address_to_miner_id = {}
        self.rank_counter = 0
        self.max_inactive_time = 20

    def register_or_update_miner(self, public_address, current_time, trigger_error=False):
        with self.lock:
            if public_address not in self.public_address_to_miner_id:
                return self.register_new_miner(public_address, current_time)
            else:
                return self.update_existing_miner(public_address, current_time, trigger_error)

    def register_new_miner(self, public_address, current_time):
        with self.lock:
            miner_id = self.rank_counter
            self.rank_counter += 1
            self.public_address_to_miner_id[public_address] = miner_id
            self.meta_miners[miner_id] = {"public_address": public_address, "first_seen": current_time, "last_seen": current_time, "total_uptime": 0}
            return miner_id

    def update_existing_miner(self, public_address, current_time, trigger_error):
        with self.lock:
            miner_id = self.public_address_to_miner_id.get(public_address)
            if miner_id is not None:
                miner_data = self.meta_miners[miner_id]
                miner_data["last_seen"] = current_time
                miner_data["total_uptime"] += current_time - miner_data["last_seen"]
                return miner_id
        return None

    def cleanup_inactive_miners(self, current_time):
        with self.lock:
            inactive_miners = [miner_id for miner_id, miner_data in self.meta_miners.items() if current_time - miner_data["last_seen"] > self.max_inactive_time]
            for miner_id in inactive_miners:
                del self.meta_miners[miner_id]
                del self.public_address_to_miner_id[self.meta_miners[miner_id]["public_address"]]
            return len(inactive_miners) > 0
