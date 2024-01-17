import json
import random
import time
from time import sleep

import bittensor as bt
import torch
from hivemind.utils.timed_storage import get_dht_time
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from bitarray import bitarray

class DatasetState:
    """
    This class shares the amount of indicies in an existing dataset for distribution among miners.
    Indices that have been used during an epoch are removed.
    (There should be a mechanism added on failure to allow for repeating)
    If the indices run out then a new epoch is calculated and the number of available indices is reset to full.
    The following
    """
    def __init__(self, dht_state, dataset_indices, run_id, default_expiration_time=600):
        assert run_id, "run_id isn't specified when run_id can't be empty/zero/null"
        self.dht_state = dht_state
        self.default_expiration_time = default_expiration_time
        self.run_id = run_id
        self.dataset_indices_original = dataset_indices
        self.dataset_indices_train = dataset_indices
        self.loss = None
        self.set_dht("dataset_indices_train", self.dataset_indices_train)
        self.set_dht("loss", self.loss)

        if self.dataset_indices_train is None:
            self.dataset_indices_train = self.get_dht("dataset_indices_train")
        if self.loss is None:
            self.loss = self.get_dht("loss")

    def _serialize_to_string(self, data_str):
        return json.dumps(data_str)

    def _deserialize_from_string(self, data_str):
        return json.loads(data_str)

    def get_dht(self, name):
        if isinstance(self.dht_state, dict):
            return self.dht_state.get(name)

    def set_dht(self, name, value):
        if isinstance(self.dht_state, dict):
            self.dht_state[name] = value

    def get_dataset_indices(self, groups_count, items_per_group):
        indices = self.get_dht("dataset_indices_train")

        no_value_flag = False

        try:
            no_value_flag = len(indices) < (groups_count * items_per_group)
        except:
            no_value_flag = True

        if no_value_flag:
            print("Ran out of dataset indices. Reloading")
            self.set_dht("dataset_indices_train", self.dataset_indices_original)
            try:
                self.epoch += 1
            except AttributeError:
                self.epoch = 1
            return self.get_dataset_indices(groups_count, items_per_group)
        
        selected_groups = []
        for _ in range(groups_count):
            search_start = random.choice(range(len(indices) - items_per_group + 1))
            start = indices.index(bitarray('0'*items_per_group), search_start)
            group = [i for i in range(start,start + items_per_group)]
            selected_groups.append(group)

            indices[group] = True

        bt.logging.info("Removing selected indices from the DHT")
        self.set_dht("dataset_indices_train", indices)
        return selected_groups

    def get_dataset_indices_test(self, batch_size):
        dataset_indices_train = self.get_dht("dataset_indices_train")
        start = random.choice(range(len(dataset_indices_train) - batch_size + 1))
        dataset_indices_test = [i for i in range(start, start + batch_size)]
        self.set_dht("dataset_indices_test", dataset_indices_test)
        # dataset_indices_train = dataset_indices_train[:start] + dataset_indices_train[start + batch_size:]
        dataset_indices_train[start:start + batch_size] = True
        self.set_dht("dataset_indices_train", dataset_indices_train)
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
