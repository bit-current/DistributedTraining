import torch
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class DatasetStateSingelton:
    '''
    This class shares the amount of indicies in an existing dataset for distribution among miners.
    Indices that have been used during an epoch are removed. 
    (There should be a mechanism added on failure to allow for repeating)
    If the indices run out then a new epoch is calculated and the number of available indices is reset to full.
    The following 
    '''
    _instance = None

    def __new__(cls, dht_state, dataset_indices,*args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatasetStateSingelton, cls).__new__(cls, *args, **kwargs)
            cls._instance._dht_state = dht_state
            cls._dataset_indices = dataset_indices
            cls.dataset_indices = dataset_indices
            
        return cls._instance

    def __getattr__(self, name):
        # Called when an attribute lookup has not found the attribute in the usual places
        return self._dht_state.get(name)

    def __setattr__(self, name, value):
        if name in ['_dht_state', '_dataset_indices']:
            # This condition is to allow initial setting of _dht_state
            super(DatasetStateSingelton, self).__setattr__(name, value)
        else:
            # Update the DHT state
            if type(value) != dict:
                self._dht_state.store(name, value)
            else:
                raise NotImplementedError("Can't use dicts directly with DHTs need to use subkey and traverse the dict")

    def get_dataset_indices(cls, m, n):
            
        """
        Selects m groups of n consecutive indices from a list in indices_dict[key].
        Each group of n indices is removed from the original list to ensure no replacement.

        :param indices_dict: Dictionary containing lists of indices.
        :param key: Key in the dictionary to access the list of indices.
        :param m: Number of groups to select.
        :param n: Number of consecutive indices in each group.
        :return: List of selected groups, each group is a list of n indices.
        """

        indices_dict = cls.dataset_indices
        if len(indices) < m * n:
            # Not enough indices to select the required number of groups"
            # Restore all the values. Then resample.

            cls.dataset_indices = cls._dataset_indices
            try:
                cls.epoch += 1
            except:
                cls.epoch = 1

            return get_dataset_indices(m,n)
            #raise ValueError()

        selected_groups = []
        for _ in range(m):
            start = random.choice(range(len(indices) - n + 1))
            group = indices[start:start + n]
            selected_groups.append(group)

            # Remove selected indices
            indices = indices[:start] + indices[start + n:]

        # Update the original list in the dictionary
        cls.dataset_indices = indices

        return selected_groups

class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name):
        if cls._instance is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance = AutoModelForCausalLM.from_pretrained(model_name).to(cls._device)
            

        return cls._instance


def upload_checkpoint(commit_message, state_averager, model, repo_path,repo_url):
    logger.info("Saving optimizer")
    torch.save(state_averager.optimizer.state_dict(), f"{repo_path}/optimizer_state.pt")
    timestamp_at_upload = time.time()
    logger.info("Started uploading to Model Hub")
    model.push_to_hub(
        repo_name=repo_path,
        repo_url=repo_url,
        commit_message=commit_message,
    )
    logger.info("Finished uploading to Model Hub")