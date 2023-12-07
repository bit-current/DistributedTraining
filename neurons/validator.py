# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 KMFODA

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


import hivemind
import time

# Bittensor
import bittensor as bt

import torch
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
#from optimum.bettertransformer import BetterTransformer
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from functools import partial
from template.utils.misc import AsyncDendritePool
from template.utils.uids import get_random_uids
from template.validator.validator_core import DatasetStateSingelton, ModelSingleton, upload_checkpoint
from template.validator import forward
from template.base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # TODO CLEAN THIS
        # Step 7: Init DHT
        self.config.initial_peers= "/ip4/161.97.156.125/tcp/8108/p2p/12D3KooWRYFcmYhf7Lyn3TH9cFxqhruhnjMYky4ehdZHFuv6M4A9"
        self.config.model_name: str = "kmfoda/tiny-random-gpt2"
        self.config.lr = 0.00001
        self.config.dataset_name = "wikitext"
        self.config.batch_size = 16
        self.config.num_of_duplicates = 2 # number of miners running the same process for validation
        self.config.run_id = '7am_run_test'
        self.config.upload_interval = 900
        self.config.weight_update_interval = 900
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dht = hivemind.DHT(initial_peers=[self.config.initial_peers], start=True)
        self.dendrite_pool = AsyncDendritePool( wallet = self.wallet, metagraph = self.metagraph )
        
        # Init Loss
        self.previous_loss = -1000
        self.latest_upload = 0
        self.latest_weight_update = 0
        self.step = 0
        print("To join the training, use initial_peers =", [str(addr) for addr in self.dht.get_visible_maddrs()])

        self.dataset = load_dataset(self.config.dataset_name, 'wikitext-2-v1', split='train')
        self.dataset_indices = [i for i in range(0, len(self.dataset))]
        self.dataset_common_state = DatasetStateSingelton(self.dht , self.dataset_indices, self.config.run_id)

        self.model = ModelSingleton.get_instance(self.config.model_name)
        self.model.to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.state_averager = TrainingStateAverager(
            dht=self.dht, 
            optimizer=partial(torch.optim.AdamW, lr=self.config.lr), 
            scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
            params=self.model.parameters(),
            start=True,
            prefix=f"{self.config.run_id}_state_averager_1",
            # state_compression=hivemind.Float16Compression(),
            # bandwidth=optimizer_args.bandwidth,
            # client_mode=optimizer_args.client_mode,
            # **asdict(averager_args),
        )
        
        # The Main Validation Loop
        bt.logging.info("Starting validator loop.")
        
    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

    async def forward(self):
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)