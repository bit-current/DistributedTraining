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

import bittensor as bt

from template.protocol import Train
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids
from template.utils.misc import AsyncDendritePool
import template


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    total_per_pass = 100000

    examples_per_uid = total_per_pass // len(miner_uids)
    # uids_indices = dataset_common_state.get_dataset_indices(len(miner_uids), examples_per_uid) #TODO add repeat on blocked
    uids_indices = [1, 2, 3]

    # TODO split queries
    queries = []
    for uid in miner_uids:
        # shuffled_uid_group = random.shuffle(uids_indices[uid_idx:uid_idx+examples_per_uid])
        #TODO get the hashes of the groups
        # Make miners return the group hashes with their losses as well.
        queries.append(
            template.protocol.Train( 
                dataset_indices = uids_indices, #TODO send different indices to different uids
                run_id = self.config.run_id,
                # initial_peers = config.initial_peers, #TODO Add a decorator or sth for this to get the values 
                batch_size = self.config.batch_size #TODO let miners decide this? Based on their hardware?
            )
        )

    # The dendrite client queries the network.
    responses = await self.dendrite_pool.async_forward(
        miner_uids,
        queries
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, uids=miner_uids, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
