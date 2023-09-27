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

import typing
from typing import Any
import bittensor as bt
import torch
from transformers import AutoModelForCausalLM

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2

from pydantic import BaseModel
# from typing import List

# # class Tensor(BaseModel):
# #     data: List[torch.FloatTensor]

# #     class Config:
# #         arbitrary_types_allowed = True

class Tensor(BaseModel):
    data: list[torch.FloatTensor] = None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = False

class Train( bt.Synapse ):
    """
    A simple Train protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    class Config:
        """
        Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = False
        arbitrary_types_allowed = True

    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Required request input hash, filled automatically when dendrite creates the request.
    # This allows for proper data validation and messages are signed with the hashes of the
    # required body fields. Ensure you have a {field}_hash field for each required field.
    dummy_input_hash: str = ""

    # Optional request output, filled by recieving axon.
    gradients: list[ bt.Tensor ] = None
    # gradients: list[torch.FloatTensor] = None

    # Optional model name
    model_name: str = "kmfoda/tiny-random-gpt2"

    # Model Weight
    # model_weights: list[ bt.Tensor ] = [layer.clone().detach() for layer in AutoModelForCausalLM.from_pretrained(model_name).parameters()]
    model_weights: str = ""
    # model_weights: list[ bt.Tensor ] = [torch.tensor(1.0), torch.tensor(1.0)]
    # [torch.tensor(1.0) for layer in AutoModelForCausalLM.from_pretrained(model_name).parameters()]
    
    # Optional learning rate
    lr: float = 1e-5
    
    # Optional dataset name
    dataset_name: str = 'wikitext'

    # Required optimizer
    optimizer_name: str = "adam"

    # Required batch size
    batch_size: int = 4

    # Optional score
    loss: float = 0

    def deserialize(self) -> int:
        """
        Deserialize the train output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of train_instance.

        Example:
        Assuming a Train instance has a dummy_output value of 5:
        >>> train_instance = Train(dummy_input=4)
        >>> train_instance.dummy_output = 5
        >>> train_instance.deserialize()
        5
        """
        return self.gradients, self.model_name, self.dataset_name, self.batch_size, self.optimizer_name, self.loss, self.model_weights
