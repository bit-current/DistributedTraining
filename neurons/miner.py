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
import pickle
import typing
from functools import partial
import bittensor as bt
import requests
from ipaddress import ip_address
# Bittensor Miner Template:
import template

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from template.utils.misc import load_wandb

import bittensor as bt
import torch
import asyncio

from datasets import load_dataset
import hivemind
from hivemind.optim.state_averager import LRSchedulerBase
from hivemind import DHT, Optimizer, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers.trainer import Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainerCallback,
    TrainingArguments,
    )

# Global variables
forward_event = False
dataset_indices = None
validation_loss = 0

class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in Optimizer.scheduler"""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        bt.logging.debug("Called NoOpScheduler.load_state_dict")

class CustomValidationCallback(TrainerCallback):
    def __init__(
        self,
        dht: DHT,
        optimizer: Optimizer,
        model: torch.nn.Module,
        trainer: Trainer,
        #local_public_key: bytes,
        #statistics_expiration: float,
        #backup_every_steps: int,
    ):
        super().__init__()
        self.model = model
        self.dht, self.optimizer = dht, optimizer
        self.trainer = trainer
        #self.local_public_key = local_public_key
        #self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        #self.backup_every_steps = backup_every_steps
        self.latest_backup = self.backup_state()
        
    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        bt.logging.info("Loading state from peers")
        self.optimizer.load_state_from_peers()
    
    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        control.should_log = True
        if not self.params_are_finite():
            self.restore_from_backup(self.latest_backup)
            return control
        
        global forward_event
        global dataset_indices
        global validation_loss
        if forward_event:
            # Select dataset indices for validation
            specific_subset_dataset = self.trainer.train_dataset.select(dataset_indices)

            # Replace the trainer's evaluation dataset
            self.trainer.eval_dataset = specific_subset_dataset

            # Run validation and capture loss
            eval_result = self.trainer.evaluate()
            validation_loss = eval_result['eval_loss']

            # Reset flag
            forward_event = False
        
        return control
    
    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> bytes:
        return pickle.dumps({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()})

    @torch.no_grad()
    def restore_from_backup(self, backup: bytes):
        state = pickle.loads(backup)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
            
            
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Init device
        self.device = self.config.neuron.device
        
        use_google_dns = True
        if use_google_dns:
            request = requests.get("https://api.ipify.org")
            request.raise_for_status()

            address = request.text
            print(f"Received public IP address of this machine: {address}")
            version = ip_address(address).version
            announce_maddrs = [f"/ip{version}/{address}/tcp/8009"]
    
        # Init DHT
        dht = hivemind.DHT(
            initial_peers=[
                "/ip4/54.80.217.105/tcp/8008/p2p/12D3KooWMn1xWT1j4zHk8pjDA9kpqp6penpFCFM7SW46JtNMunKi"], 
            host_maddrs=[f"/ip4/0.0.0.0/tcp/8009", 
                        f"/ip4/0.0.0.0/udp/8009/quic"],
            announce_maddrs = announce_maddrs,
            start=True)
        
        utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=True)

        # Init model
        self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)
        
        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Set up a decentralized optimizer that will average with peers in background
        opt = torch.optim.AdamW(self.model.parameters(), lr = self.config.neuron.lr)
        optimizer = hivemind.Optimizer(
            dht=dht,                    # use a DHT that is connected with other peers
            run_id=self.config.neuron.run_id,        # unique identifier of this collaborative run
            scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
            batch_size_per_step=1,     # each call to opt.step adds this many samples towards the next epoch
            target_batch_size=10,    # after peers collectively process this many samples, average weights and begin the next epoch
            optimizer=opt,              # wrap the SGD optimizer defined above
            use_local_updates=True,     # perform optimizer steps with local gradients, average parameters in background
            matchmaking_time=10.0,       # when averaging parameters, gather peers in background for up to this many seconds
            averaging_timeout=15.0,     # give up on averaging if not successful in this many seconds
            verbose=True                # print logs incessently
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        dataset = load_dataset(self.config.neuron.dataset_name, 'wikitext-2-v1', split='train')
        
        # Encode the dataset
        encoded_dataset = dataset.map(self.encode, batched=True)
        
        
        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            #args=training_args,
            train_dataset=encoded_dataset,
            eval_dataset=encoded_dataset,
            optimizers=(optimizer, NoOpScheduler(optimizer)),
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
            )
        
        self.trainer.add_callback(
            CustomValidationCallback(
                    dht,
                    optimizer,
                    self.model,
                    self.trainer
                )
            )
        #self.trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
        #self.trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
        

    def encode(self, examples):
        # Tokenize the text
        tokenized_examples = self.tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

        # For language modeling tasks, the labels are usually the same as the input_ids
        tokenized_examples['labels'] = tokenized_examples['input_ids'].copy()

        return tokenized_examples

    # # Define encoding function
    # def encode(self, examples):
    #     return self.tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')


    async def forward(
        self, synapse: template.protocol.Train
    ) -> template.protocol.Train:
        """
        Processes the incoming 'Train' synapse by performing a training run

        Args:
            synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

        Returns:
            template.protocol.Train: The synapse object with the 'loss' field set to models loss.
        """
        global forward_event
        forward_event = True
        global dataset_indices
        dataset_indices = synapse.dataset_indices
        if not self.config.neuron.dont_wandb_log:
            self.wandb.log({"received_indices": synapse.dataset_indices})
        
        # Wait for the validation to complete inside on_step_end inside CustomValidationCallback
        while forward_event:
            await asyncio.sleep(0.1)  # Avoid busy waiting

        global validation_loss
        synapse.loss = validation_loss

        bt.logging.info(f"Final Loss: {synapse.loss}")
        
        return synapse


    async def blacklist(
        self, synapse: template.protocol.Train
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Train): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        hotkey = synapse.dendrite.hotkey
        synapse_type = type(synapse).__name__

        uid = None
        axon = None
        for _uid, _axon in enumerate(self.metagraph.axons):
            if _axon.hotkey == hotkey:
                uid = _uid
                axon = _axon
                break

        if uid is None:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
            )
            return (
                True,
                f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
            )

        if self.config.blacklist.force_validator_permit and (not self.config.blacklist.allow_non_registered):
            # Check stake if uid is recognize
            tao = self.metagraph.neurons[uid].stake.tao
            if tao < self.config.neuron.vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {tao} < {self.config.neuron.vpermit_tao_limit} from {hotkey}",
                )

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: template.protocol.Train) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Train): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        # Start training
        miner.trainer.train()
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
