import random
import time

import bittensor as bt
import torch
from hivemind.utils.timed_storage import get_dht_time

from bitarray import bitarray

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
        self.loss = None
        self.epoch = 0
        self.set_dht("dataset_indices_train", self.dataset_indices_train)
        self.set_dht("loss", self.loss)
        self.set_dht("epoch", self.epoch)
        
        if self.dataset_indices_train is None:
            self.dataset_indices_train = self.get_dht("dataset_indices_train")
        if self.loss is None:
            self.loss = self.get_dht("loss")
        if self.epoch is None:
            self.epoch = self.get_dht("epoch")
   
    def get_dht(self, name):
        try:
            value, expiration = self.dht.get(name, latest=True)
            if value is not None:
                return value
        except Exception as e:
            print(f"Error accessing DHT: {e}")
        return None

    def set_dht(self, name, value):
        try:
            expiration_time = get_dht_time() + self.default_expiration_time
            store_ok = self.dht.store(name, value, expiration_time=expiration_time)
            if not store_ok:
                print(f"Failed to store key: {name}")
        except Exception as e:
            print(f"Error storing to DHT: {e}")
    
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

    # def update_step(cls):
    #     step = cls.get_dht("step")
    #     cls.set_dht("step", step + 1)


def upload_checkpoint(commit_message, state_averager, model, repo_path, repo_url):
    bt.logging.info("Saving optimizer")
    torch.save(state_averager.optimizer.state_dict(), f"{repo_path}/optimizer_state.pt")
    timestamp_at_upload = time.time()
    bt.logging.info("Started uploading to Model Hub")
    model.push_to_hub(
        repo_name=repo_path,
        repo_url=repo_url,
        commit_message=commit_message,
    )
    bt.logging.info("Finished uploading to Model Hub")

if __name__ == "__main__":
    import hivemind
    # Initialize the dummy data and the DatasetState instance
    dataset_length = 96015  # Reduced size for testing
    dataset_indices = bitarray('0' * dataset_length)  # Initialize all indices to 0 (unused)
    #dataset_dict = {}  # Dummy DHT, aka dht_state
    dht = hivemind.DHT(start=True)
    run_id = "dummy_run_id"
   
    dataset_state = DatasetState(dht, dataset_indices, run_id)
    
    # Test get_dataset_indices
    selected_groups = dataset_state.get_dataset_indices(2, 500)
    print("Selected groups for training:", selected_groups)
    print(dataset_state.dataset_indices_train.count(0))
    selected_groups = dataset_state.get_dataset_indices(2, 500)
    print("Selected groups for training:", selected_groups)
    print(dataset_state.dataset_indices_train.count(0))

    # Test get_dataset_indices_test with DHT
    test_indices = dataset_state.get_dataset_indices_test(20)
    print(test_indices)
    print(dht.get("dataset_indices_test"))
    dht.shutdown()