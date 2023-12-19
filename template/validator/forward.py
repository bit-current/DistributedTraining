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
from template.utils.misc import AsyncDendritePool
import template
import asyncio


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    
    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"uids":self.miner_uids,
                    "dataset_indices":self.dataset_indices_list})

    query_tasks = []

    queries = []
    for uid, uid_dataset in zip(self.miner_uids, self.dataset_indices_list):
        #TODO get the hashes of the groups
        # Make miners return the group hashes with their losses as well.
        queries.append(
            template.protocol.Train( 
                dataset_indices = uid_dataset,
                run_id = self.config.neuron.run_id,
                batch_size = self.config.neuron.batch_size_train  #TODO let miners decide this? Based on their hardware. Then reconcile if needed?
            )
        )

    # The dendrite client queries the network.
    query_tasks.append(
        self.dendrite_pool.async_forward(
            self.miner_uids,
            queries
        )
    )
    responses = await asyncio.gather(*query_tasks)
    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {[{'Loss':response.loss,'Dataset Indices':(min(response.dataset_indices), max(response.dataset_indices)), 'IP':response.dendrite.ip, 'Port':response.dendrite.port, 'Hotkey':response.dendrite.hotkey} for response in responses[0] if response.dendrite.status_code == 200 ]}")
    
    return responses
