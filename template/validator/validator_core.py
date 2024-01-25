import random
from bitarray import bitarray
from hivemind.utils.timed_storage import get_dht_time
import bittensor as bt

class DatasetState:
    """
    This class shares the amount of indicies in an existing dataset for distribution among miners.
    Indices that have been used during an epoch are removed.
    (There should be a mechanism added on failure to allow for repeating)
    If the indices run out then a new epoch is calculated and the number of available indices is reset to full.
    The following
    """
    def __init__(self, dht, dataset_indices, run_id, default_expiration_time=600):
        assert run_id, "run_id isn't specified when run_id can't be empty/zero/null"
        self.dht = dht
        self.default_expiration_time = default_expiration_time
        self.run_id = run_id
        self.dataset_indices_original = dataset_indices
        self.dataset_indices_train = dataset_indices
        self.loss = self.get_dht("loss")
        self.epoch = self.get_dht("epoch")
        self.step = self.get_dht("step")

        if self.step is None:
            self.step = 0
            self.set_dht("step", self.step)
        
        if self.epoch is None:
            self.epoch = 0
            self.set_dht("epoch", self.epoch)
   
    def get_dht(self, name):
        try:
            value = self.dht.get(name, latest=True)
            if value is not None:
                return value.value
        except Exception as e:
            bt.logging.error(f"Error accessing DHT: {e}")
        return None

    def set_dht(self, name, value):
        try:
            expiration_time = get_dht_time() + self.default_expiration_time
            store_ok = self.dht.store(name, value, expiration_time=expiration_time)
            if not store_ok:
                bt.logging.error(f"Failed to store key: {name}")
        except Exception as e:
            bt.logging.error(f"Error storing to DHT: {e}")
    
    def get_dataset_indices(self, groups_count, items_per_group):
        
        if self.dataset_indices_train.count(0) < groups_count * items_per_group:
            print("Ran out of dataset indices. Resetting")
            self.dataset_indices_train.setall(0)
            self.epoch += 1
            self.set_dht("epoch", self.epoch)
            return self.get_dataset_indices(groups_count, items_per_group)
        
        selected_groups = []
        for _ in range(groups_count):
            search_start = random.choice(range(len(self.dataset_indices_train) - items_per_group + 1))
            start = self.dataset_indices_train.index(bitarray('0'*items_per_group), search_start)
            group = [i for i in range(start,start + items_per_group)]
            selected_groups.append(group)

            self.dataset_indices_train[group] = True

        return selected_groups

    def get_dataset_indices_test(self, batch_size):
        start = random.choice(range(len(self.dataset_indices_train) - batch_size + 1))
        dataset_indices_test = [i for i in range(start, start + batch_size)]
        self.set_dht("dataset_indices_test", dataset_indices_test)
        self.dataset_indices_train[start:start + batch_size] = True

        return dataset_indices_test

    def update_step(self):
        self.step = self.get_dht("step")
        if self.step is not None:
            self.set_dht("step", self.step + 1)
        else:
            return None