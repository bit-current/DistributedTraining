# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
from typing import List

import bittensor as bt
import torch

def get_loss(self, dataset_indices):

    dataset_sample = self.dataset.select(dataset_indices)

    # Encode the dataset
    encoded_dataset = dataset_sample.map(self.encode, batched=True)

    # Move batch to device
    input_ids = torch.tensor(encoded_dataset["input_ids"]).to(self.device)
    attention_mask = torch.tensor(encoded_dataset["attention_mask"]).to(self.device)
    labels = torch.tensor(encoded_dataset["input_ids"]).to(self.device)

    # Forward pass
    outputs = self.model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )

    # Backward pass
    loss = outputs.loss

    return loss

def get_local_score(self, synapse):

    loss = get_loss(self, synapse.dataset_indices[-synapse.batch_size:])
    # The miner's local score is the variance between the loss it returns and the 
    # loss the validator calculates for the last batch of data sent to that miner
    score = 1-(abs(loss-synapse.loss)/loss)

    return score
    

def get_rewards(
    self,
    uids: List[int],
    responses: list
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """

    load_state_from_peers_status = False
    retries = 0
    while load_state_from_peers_status is False:
        try:
            load_state_from_peers_status = self.state_averager.load_state_from_peers()
        except Exception as e:
            bt.logging.error(f"Attempt {retries + 1} to write to the load state from peers failed: {e}")
            retries += 1
            bt.logging.error(f"Retrying ...")

    # self.global_step = self.dataset_common_state.get_dht("step")
    # if self.global_step % 100 == 0:
    #     self.dataset_indices_list_test = (
    #         self.dataset_common_state.get_dataset_indices_test(
    #             self.config.neuron.batch_size_test
    #         )
    #     )
    if (self.step % 100 == 0) and (self.step != 0):
        self.dataset_indices_list_test = self.dataset_common_state.get_dataset_indices_test(self.config.neuron.local_batch_size_test)

    # Get loss on randomly selected test dataset to be used for the Global Score
    loss = get_loss(self, self.dataset_indices_list_test)

    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"loss": loss, "previous_loss": self.previous_loss})

    # Get latest previous loss from DHT
    # self.previous_loss = self.dataset_common_state.get_dht("loss")
    self.previous_loss = self.dataset_common_state.get_dht("loss")
    bt.logging.info(f"Previous Global Loss:    {self.previous_loss}")
    bt.logging.info(f"Current Global Loss:     {loss}")

    # Compute Global Score
    if ((self.previous_loss is None) or ((self.previous_loss - loss) < 0)) and (responses[0] != []):
        score = 1
        # self.dataset_common_state.set_dht("loss", float(loss))
        self.dataset_common_state.set_dht("loss", loss)
    else:
        score = 0.1 #Training has stagnated

    # Log score, previous and current loss
    bt.logging.info(f"Global Score:            {score}")

    # Set previous loss to current loss
    self.previous_loss = loss
    
    # Get all the reward results by iteratively calling your reward() function.
    scores = torch.FloatTensor([score if response.dendrite.status_code == 200 and response.loss != [] and response.loss != -1 else 0.1 
                                for _, response in zip(uids, responses[0])]).to(self.device)
    bt.logging.info(f"Global Scores: {scores}")

    # Adjust Global Score with Local Score
    test_uids_index = [uid_index for uid_index, uid in enumerate(uids) 
                       if responses[0][uid_index].dendrite.status_code == 200]
    
    test_uids_sample_index = random.sample(test_uids_index, k = min(4, len(test_uids_index)))
    
    scores = torch.FloatTensor([scores[uid_index] * get_local_score(self, responses[0][uid_index]) 
                                if uid_index in test_uids_sample_index else scores[uid_index] 
                                for uid_index,_ in enumerate(uids)]).to(self.device)
    
    bt.logging.info(f"Adjusted Global Scores: {scores}")
    
    return scores

