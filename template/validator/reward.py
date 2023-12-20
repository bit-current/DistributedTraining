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


def reward(query: int, response: int) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """

    return 1.0 if response == query * 2 else 0


def get_rewards(
    self,
    uids: List[int],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    try:
        self.state_averager.load_state_from_peers()
    except Exception as e:
        breakpoint()

    global_step = self.dataset_common_state.get_dht("step")
    if global_step % 100 == 0:
        self.dataset_indices_list_test = (
            self.dataset_common_state.get_dataset_indices_test(
                self.config.neuron.batch_size_test
            )
        )

    # Select the correct datapoints
    dataset_sample = self.dataset.select(self.dataset_indices_list_test)

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

    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"loss": loss, "previous_loss": self.previous_loss})

    # Get latest previous loss from DHT
    self.previous_loss = self.dataset_common_state.get_dht("loss")
    bt.logging.info(f"Previous loss:    {self.previous_loss}")
    bt.logging.info(f"Current loss:     {loss}")

    # Compute score
    if (self.previous_loss is None) or ((loss - self.previous_loss) > 0):
        score = 1
        self.dataset_common_state.set_dht("loss", float(loss))
    else:
        score = 0

    # Log score, previous and current loss
    bt.logging.info(f"Score:            {score}")

    # Set previous loss to current loss
    self.previous_loss = loss

    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor([score for _ in uids]).to(self.device)
