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
from template.data.dataset import SubsetFalconLoader

def get_loss(self, dataset_indices, batch_size, gradient_accumilation_steps):

    # Create Dataloader
    dataloader = SubsetFalconLoader(
        batch_size=batch_size, sequence_length=1024, rows=dataset_indices
    )

    total_loss = 0
    n_acc_steps = 0
    accumulation_steps = gradient_accumilation_steps

    # Train data for one epoch
    for step, batch in enumerate(dataloader):

        inputs = batch.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=inputs, labels=inputs)
        
        # Normalize loss to account for batch accumulation
        loss = outputs.loss / accumulation_steps  

        # Backward Pass
        loss.backward()

        # Accumulate Total Loss
        total_loss += outputs.loss.detach().item() 

        if (step + 1) % accumulation_steps == 0:
            n_acc_steps += 1
            self.opt.step()         # Adjust gradient
            self.opt.zero_grad()    # Clear gradients
            
            bt.logging.info(f"Step {n_acc_steps} Loss: {outputs.loss.detach().item()}")
            
            if not self.config.neuron.dont_wandb_log:
                self.wandb.log({"loss": outputs.loss.detach().item(), "opt_local_epoch": self.opt.local_epoch})

        torch.cuda.empty_cache()

    average_loss = total_loss / step

    bt.logging.info(f"Final Loss: {outputs.loss.detach().item()}")
    bt.logging.info(f"Average Loss: {average_loss}")

    return average_loss

def get_local_score(self, synapse):
    if self.opt.tracker.global_progress.epoch != self.current_epoch:
        score = 1
        bt.logging.info(f"Skipping local scoring as epoch has changed from {self.current_epoch} to {self.opt.tracker.global_progress.epoch}")
    else:
        loss = get_loss(self, synapse.dataset_indices, synapse.batch_size, synapse.gradient_accumilation_steps)
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
    from hivemind.p2p import P2P, P2PContext, P2PDaemonError, P2PHandlerError, PeerID, ServicerBase
    from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time
    key_manager = self.opt.state_averager.matchmaking_kwargs
    prefix = key_manager["prefix"]
    peer_priority, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or ({}, None)
    peer_priority = {PeerID(peer_id): (float(info.value), random.random()) for peer_id, info in peer_priority.items() if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))}
    for peer in peer_priority: break
    breakpoint()
    self._p2p = await self.dht.replicate_p2p()

    stub = self.opt.state_averager.get_stub(self.opt.state_averager._p2p, peer, namespace=self.opt.state_averager.prefix)
    # stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
    # current_tensor_parts, tensors = [], []

    # # TODO merge this with hivemind.compression.deserialize_tensor_stream
    # async for message in aiter_with_timeout(stream, timeout=timeout):
    #     if message.metadata:
    #         metadata = self.serializer.loads(message.metadata)
    #     if message.tensor_part.dtype and current_tensor_parts:
    #         # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
    #         tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
    #         current_tensor_parts = []
    #     current_tensor_parts.append(message.tensor_part)
    # if current_tensor_parts:
    #     tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

    
    breakpoint()
    retries = 0
    while load_state_from_peers_status is False:
        try:
            load_state_from_peers_status = self.opt.state_averager.load_state_from_peers()
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
    loss = get_loss(self, self.dataset_indices_list_test, self.config.neuron.local_batch_size_test, self.config.neuron.local_gradient_accumilation_steps_test)

    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"loss": loss, "previous_loss": self.previous_loss})

    # Get latest previous loss from DHT
    # self.previous_loss = self.dataset_common_state.get_dht("loss")
    self.previous_loss = self.dataset_common_state.get_dht("loss")
    bt.logging.info(f"Previous Global Loss:    {self.previous_loss}")
    bt.logging.info(f"Current Global Loss:     {loss}")

    # Compute Global Score
    if ((self.previous_loss is None) or ((loss - self.previous_loss) < 0)) and (responses[0] != []):
        score = 1
        # self.dataset_common_state.set_dht("loss", float(loss))
        self.dataset_common_state.set_dht("loss", loss)
    else:
        score = 0

    # Log score, previous and current loss
    bt.logging.info(f"Global Score:            {score}")

    # Set previous loss to current loss
    self.previous_loss = loss
    
    # Get all the reward results by iteratively calling your reward() function.
    scores = torch.FloatTensor([score if response.dendrite.status_code == 200 and response.loss != [] else 0 
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

