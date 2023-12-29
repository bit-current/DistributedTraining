import hivemind
import time
import pickle
import typing
from functools import partial
import bittensor as bt

import bittensor as bt
import torch
from datasets import load_dataset
import hivemind
from hivemind.optim.state_averager import LRSchedulerBase
from hivemind import DHT, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
from ipaddress import ip_address

import transformers
from transformers.trainer import Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainerCallback,
    TrainingArguments,
    )

# Global variable
forward_event = False

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
        #local_public_key: bytes,
        #statistics_expiration: float,
        #backup_every_steps: int,
    ):
        super().__init__()
        self.model = model
        self.dht, self.optimizer = dht, optimizer
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
        if forward_event:
            # Select dataset indices for validation
            specific_subset_dataset = self.dataset.select(self.dataset_indices)

            # Replace the trainer's evaluation dataset
            self.trainer.eval_dataset = specific_subset_dataset

            # Run validation and capture loss
            eval_result = self.trainer.evaluate()
            self.validation_loss = eval_result['eval_loss']

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
            
            


def encode(examples):
    # Tokenize the text
    tokenized_examples = tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

    # For language modeling tasks, the labels are usually the same as the input_ids
    tokenized_examples['labels'] = tokenized_examples['input_ids'].copy()

    return tokenized_examples


# This is the main function, which runs the miner.
if __name__ == "__main__":
    
    # Init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_google_dns = True
    if use_google_dns:
        request = requests.get("https://api.ipify.org")
        request.raise_for_status()

        address = request.text
        print(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        announce_maddrs = [f"/ip{version}/{address}/tcp/8009"]
    
    # Init DHT and model
    dht = hivemind.DHT(
        initial_peers=[
            "/ip4/54.80.217.105/tcp/8008/p2p/12D3KooWMn1xWT1j4zHk8pjDA9kpqp6penpFCFM7SW46JtNMunKi"], 
        host_maddrs=[f"/ip4/0.0.0.0/tcp/8009", 
                     f"/ip4/0.0.0.0/udp/8009/quic"],
        announce_maddrs = announce_maddrs,
        start=True)
    model = AutoModelForCausalLM.from_pretrained("kmfoda/tiny-random-gpt2")
    
    # Move the model to the appropriate device
    model = model.to(device)

    # Set up a decentralized optimizer that will average with peers in background
    opt = torch.optim.AdamW(model.parameters(), lr = 0.00001)
    optimizer = hivemind.Optimizer(
        dht=dht,                    # use a DHT that is connected with other peers
        run_id="funnybizz",        # unique identifier of this collaborative run
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
        batch_size_per_step=4,     # each call to opt.step adds this many samples towards the next epoch
        target_batch_size=32,    # after peers collectively process this many samples, average weights and begin the next epoch
        optimizer=opt,              # wrap the SGD optimizer defined above
        use_local_updates=True,     # perform optimizer steps with local gradients, average parameters in background
        matchmaking_time=10.0,       # when averaging parameters, gather peers in background for up to this many seconds
        averaging_timeout=15.0,     # give up on averaging if not successful in this many seconds
        verbose=True                # print logs incessently
    )
    
    tokenizer = AutoTokenizer.from_pretrained("kmfoda/tiny-random-gpt2")
    # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("wikitext", 'wikitext-2-v1', split='train')
    
    # Encode the dataset
    encoded_dataset = dataset.map(encode, batched=True)
    
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        #args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
        optimizers=(optimizer, NoOpScheduler(optimizer)),
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks=[
            CustomValidationCallback(
                dht,
                optimizer,
                model
            )
        ],
    )
    
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    
    trainer.train()