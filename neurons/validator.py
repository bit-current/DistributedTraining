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


import asyncio
import re
import time
from functools import partial
from ipaddress import ip_address

import bittensor as bt
import hivemind
import requests
import torch
import wandb
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.optim.progress_tracker import ProgressTracker
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from template.base.validator import BaseValidatorNeuron
from template.utils.misc import AsyncDendritePool, load_wandb, setup_logging
from template.validator import forward
from template.validator.validator_core import DatasetState
from bitarray import bitarray


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.init_dht()

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(self.config, self.wallet, "validator", str(self.dht.peer_id))

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool(
            wallet=self.wallet, metagraph=self.metagraph
        )

        # # Init Dataset
        dataset_length = 968000015
        self.dataset_indices = bitarray(dataset_length)
        # self.dataset_dict = dict()  # Init a dict to use as placeholder DHT

        self.dataset_common_state = DatasetState(
            self.dht, self.dataset_indices, self.config.neuron.run_id
        )
        
        # self.dataset_indices_list_test = self.dataset_common_state.get_dht("dataset_indices_train")
        # if self.dataset_indices_list_test is None:
        #     self.dataset_indices_list_test = self.dataset_common_state.get_dht("dataset_indices_test")
        self.dataset_indices_list_test = (
            self.dataset_common_state.get_dataset_indices_test(
                self.config.neuron.local_batch_size_test * self.config.neuron.local_gradient_accumilation_steps_test
            )
        )

        self.global_step = self.dataset_common_state.get_dht("step")
        if self.global_step is None:
            self.global_step = 0
            # self.dataset_common_state.set_dht("step")

        # Init Loss
        self.previous_loss = self.dataset_common_state.get_dht("loss")
        self.latest_upload = 0
        self.latest_weight_update = 0
        self.step = 0
        self.global_step = self.dataset_common_state.get_dht("step")

        # Init device
        self.device = self.config.neuron.device

        # Init Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.neuron.model_name
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Init State Averager
        self.state_averager = TrainingStateAverager(
            dht=self.dht,
            optimizer=partial(torch.optim.AdamW, lr=self.config.neuron.lr),
            scheduler=None,
            params=self.model.parameters(),
            allow_state_sharing=False,
            start=True,
            prefix=f"{self.config.neuron.run_id}_state_averager",
            state_compression=hivemind.Float16Compression(),
            # **asdict(averager_args),
        )
        
        # Get Current Epoch
        self.current_epoch = 1 # Dummy fix need to swithc to self.opt.tracker.global_progress.epoch
        
        # Start Main Validation Loop
        bt.logging.info("Starting validator loop.")

    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

    def init_dht(self):
        # Init DHT
        if self.config.dht.use_google_dns:
            request = requests.get("https://api.ipify.org")
            request.raise_for_status()

            address = request.text
            bt.logging.info(f"Received public IP address of this machine: {address}")
            version = ip_address(address).version
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]
        else:
            version = "4"
            address = self.config.dht.announce_ip
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]

        # Init list of available DHT addresses from wandb
        api = wandb.Api()
        initial_peers_list = self.config.neuron.initial_peers
        runs = api.runs(
            f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project}"
        )
        for ru in runs:
            if ru.state == "running":
                for peer in ru.config["neuron"]["initial_peers"]:
                    if peer not in initial_peers_list:
                        initial_peers_list.append(peer)

        retries = 0
        while retries <= len(initial_peers_list):
            if retries == len(initial_peers_list):
                raise Exception("Max retries reached, operation failed.")
            try:
                # Init DHT
                self.dht = hivemind.DHT(
                    host_maddrs=[
                        f"/ip4/0.0.0.0/tcp/{self.config.dht.port}",
                        f"/ip4/0.0.0.0/udp/{self.config.dht.port}/quic",
                    ],
                    initial_peers=[initial_peers_list[retries]],
                    announce_maddrs=announce_maddrs,
                    start=True,
                    # client_mode = True,
                )
                bt.logging.info(
                    f"Successfully initialised dht using initial_peer as {initial_peers_list[retries]}"
                )
                break
            except Exception as e:
                bt.logging.error(
                    f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peers_list[retries]} failed with error: {e}"
                )
                retries += 1
                bt.logging.error(f"Retrying...")

        # Write local dht address to config
        self.config.neuron.initial_peers = self.config.neuron.initial_peers + [
            re.sub("ip4/?(.*?)/", f"ip{version}/{address}/", str(addr), flags=re.DOTALL)
            for addr in self.dht.get_visible_maddrs()
        ]

    async def forward(self):
        return await forward(self)


# # The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    setup_logging()
    with Validator() as validator:
        while True:
            time.sleep(5)
