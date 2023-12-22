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

# Otherwise just use time.time, as per here: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/utils/timed_storage.py#L12


class DatasetStateSingelton:
    """
    This class shares the amount of indicies in an existing dataset for distribution among miners.
    Indices that have been used during an epoch are removed.
    (There should be a mechanism added on failure to allow for repeating)
    If the indices run out then a new epoch is calculated and the number of available indices is reset to full.
    The following
    """

    _instance = None

    def __new__(
        cls,
        dht_state,
        dataset_indices,
        run_id,
        default_expiration_time=600,
        *args,
        **kwargs,
    ):
        if not cls._instance:
            cls._instance = super(DatasetStateSingelton, cls).__new__(
                cls, *args, **kwargs
            )
            cls._instance.dht_state = dht_state
            cls.default_expiration_time = default_expiration_time
            assert run_id, "run_id isn't specified when run_id can't be empty/zero/null"
            cls.run_id = run_id
            cls.dataset_indices_original = dataset_indices
            cls.dataset_indices_train = cls._instance.get_dht("dataset_indices_train")
            cls.loss = cls._instance.get_dht("loss")
            cls.step = cls._instance.get_dht("step")
            if cls.step is None:
                cls.step = 0
                cls._instance.set_dht("step", cls.step)

        return cls._instance

    @staticmethod
    def _serialize_to_string(data_str):
        """
        Serializes the dataset indices to a string.
        """
        # Assuming dataset_indices is a list or a similar serializable structure
        return json.dumps(data_str)

    @staticmethod
    def _deserialize_from_string(data_str):
        """
        Deserializes the string back to dataset indices.
        """
        # Assuming the data_str is in JSON format
        return json.loads(data_str.value)

    def get_dht(cls, name, max_retries = 10, base_delay = 1)):
        sleep(2)
        
        retries = 0
        while retries < max_retries:
            try:
                stored_data = cls.dht_state.get(f"{cls.run_id}_{name}")
                return cls._deserialize_from_string(stored_data) if stored_data else None
            except Exception as e:
                bt.logging.error(f"Attempt {retries + 1} to read from the DHT failed: {e}")
                retries += 1
                delay = (base_delay * 2 ** retries + random.uniform(0, 1))
                bt.logging.error(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception("Max retries reached, operation failed.")

       
    def set_dht(cls, name, value, max_retries = 10, base_delay = 1):
        sleep(2)
        serialized_value = cls._serialize_to_string(value)

        retries = 0
        while retries < max_retries:
            try:
                status = cls.dht_state.store(
                    f"{cls.run_id}_{name}",
                    serialized_value,
                    get_dht_time() + cls.default_expiration_time,
                )
                return status
            except Exception as e:
                bt.logging.error(f"Attempt {retries + 1} to write to the DHT failed: {e}")
                retries += 1
                delay = (base_delay * 2 ** retries + random.uniform(0, 1))
                bt.logging.error(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception("Max retries reached, operation failed.")



    def get_dataset_indices(cls, groups_count, items_per_group):
        """
        Selects m groups of n consecutive indices from a list in indices_dict[key].
        Each group of n indices is removed from the original list to ensure no replacement.

        :param indices_dict: Dictionary containing lists of indices.
        :param key: Key in the dictionary to access the list of indices.
        :param groups_count: Number of groups to select.
        :param items_per_group: Number of consecutive indices in each group.
        :return: List of selected groups, each group is a list of n indices.
        """
        indices = cls.get_dht("dataset_indices_train")
        no_value_flag = False

        try:
            no_value_flag = len(indices) < (groups_count * items_per_group)
        except:
            no_value_flag = True

        if no_value_flag:
            bt.logging.info("Ran out of dataset indices. Reloading")
            # Not enough indices to select the required number of groups"
            # Restore all the values. Then resample.
            cls.set_dht("dataset_indices_train", cls.dataset_indices_original)
            try:
                cls.epoch += 1
            except:
                cls.epoch = 1

            return cls.get_dataset_indices(groups_count, items_per_group)

        selected_groups = []
        for _ in range(groups_count):
            start = random.choice(range(len(indices) - items_per_group + 1))
            group = indices[start : start + items_per_group]
            selected_groups.append(group)

            # Remove selected indices
            indices = indices[:start] + indices[start + items_per_group :]

        # Update the original list in the dictionary
        bt.logging.info("Removing selected indices from the DHT")
        cls.set_dht("dataset_indices_train", indices)

        return selected_groups

    def get_dataset_indices_test(cls, batch_size):
        """
        Selects m groups of n consecutive indices from a list in indices_dict[key].
        Each group of n indices is removed from the original list to ensure no replacement.

        :return: List of selected groups, each group is a list of n indices.
        """
        dataset_indices_train = cls.get_dht("dataset_indices_train")

        # Select test indices
        start = random.choice(range(len(dataset_indices_train) - batch_size + 1))
        dataset_indices_test = dataset_indices_train[start : start + batch_size]
        cls.set_dht("dataset_indices_test", dataset_indices_test)

        # Remove test indices from train indices
        dataset_indices_train = (
            dataset_indices_train[:start] + dataset_indices_train[start + batch_size :]
        )
        cls.set_dht("dataset_indices_train", dataset_indices_train)

        return dataset_indices_test

    def update_step(cls):
        step = cls.get_dht("step")
        cls.set_dht("step", step + 1)


class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name, device):
        if cls._instance is None:
            cls._instance = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        return cls._instance


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
