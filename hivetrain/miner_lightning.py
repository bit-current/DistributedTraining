import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelPruning,
    StochasticWeightAveraging,
)
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback, ProgressBar, TQDMProgressBar
from lightning_hivemind.strategy import HivemindStrategy
from functools import partial
from tqdm.auto import tqdm
import psutil
from math import isnan

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from datasets import load_dataset
import ipaddress
import re
import logging
import argparse

import requests
import time
from lightning.pytorch.callbacks import Callback
from hivetrain.config import Configurator
from hivetrain.btt_connector import get_validator_uids_and_addresses, BittensorNetwork, serve_axon



logging.getLogger("lightning.pytorch").setLevel(logging.INFO)


config = Configurator.combine_configs()

BittensorNetwork.initialize(config)

# Now you can access wallet, subtensor, and metagraph like this:
wallet = BittensorNetwork.wallet
subtensor = BittensorNetwork.subtensor
metagraph = BittensorNetwork.metagraph

# capture arguments passed to this python script
#parser = argparse.ArgumentParser(description="Get configs from arguments to this script.")
#parser.add_argument('--initial_peers', type=list, help='Your peer. Use --initial_peers multiple times to pass multiple peers.', default=[], nargs='?')
#args = parser.parse_args()
#TODO add a method to provide 
initial_peers = config.neuron.initial_peers#args.initial_peers

# initialized and load the model
block_size = 1024 #I think this is the context length
config = AutoConfig.from_pretrained(
    "gpt2",
    n_emb=256,
    n_layer=3,
    n_head=3,
    n_inner=1024,
    torch_dtype=torch.float32
)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2", 
    cache_dir="/tmp/tokenizer",
    padding="max_length",
    padding_side="left",
    use_fast=True,
    return_overflowing_tokens=True,
    truncation=True,
)
tokenizer.pad_token = tokenizer.eos_token

dataset_config = {
    "dataset": "tiiuae/falcon-refinedweb",
    "key": "content",
    "split": "train",
    "block_size": block_size,
}

# create a datamodule to wrap our remote datasets
class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.train_data = None
        self.train_data = StreamingDataset(
            self.tokenizer, config
        )
    ## TODO can we replace with DistributedDataLoader
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=1,
            pin_memory=True,
            num_workers=2,
        )

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, conf):
        self.tokenizer = tokenizer
        self.config = conf
        self.dataset = load_dataset(
            self.config.get("dataset", "tiiuae/falcon-refinedweb"),
            split=self.config.get("split", "train"),
            streaming=True,
            cache_dir="/tmp/pile"
        )

    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=1000,
        )

        block_size = self.config.get("block_size", 512)

        batch = []
        for document in shuffled:
            tokenized = self.tokenizer(
                text=document.get(self.config.get("key", "default")),
                max_length=block_size,
                stride=0,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="np",
            )["input_ids"]
            choice = random.choice(tokenized)
            if len(choice) == 0:
                continue
            elif len(batch) == 0:
                batch = choice
            else:
                np.append(batch, self.tokenizer.eos_token_id)
                batch = np.concatenate([batch, choice])
            if len(batch) >= block_size:
                yield batch[:block_size]
                batch = []
            else:
                continue

# prepare a dataset for use with training
dataset = StreamingDataModule(tokenizer, dataset_config)

# wrap the LightningModule in a custom class
class MinerTrainer(LightningModule):
    """
    A training module for AIGen.
    """

    def __init__(self, model, optimizer, tokenizer, hparams):
        super(MinerTrainer, self).__init__()

        self.model, self.optimizer, self.tokenizer = (
            model,
            optimizer,
            tokenizer,
        )
        self.automatic_optimization = True
        self.pbar = None
        self.save_hyperparameters(hparams)

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0] #FIXME am i training? Check parent class
        #TODO Am I a good spot for communicating to vali?
        self.log(
            "train_loss", float(loss), on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def on_train_batch_end(self, trainer, lm, outputs):
        schedule = self.lr_schedulers()
        step = self.global_step

        if hasattr(schedule, "current_step"):
            step = schedule.current_step
        elif hasattr(self.trainer.strategy.optimizers[0], "local_epoch"):
            step = self.trainer.strategy.optimizers[0].local_epoch

        self.log("step", int(step), on_step=True, on_epoch=False, sync_dist=True)

        if hasattr(schedule, "step"):
            schedule.step()

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return [self.optimizer]

# define the model hyperparameters
hparams = dict(
    learning_rate=0.001,
    weight_decay=0.1,
    eps=1e-8,
    warmup_steps=0,
    batch_size=1,
    num_steps=10000,
    target_batch_size=512,
    block_size=block_size
)

# define the hivemind strategy
strategy = HivemindStrategy(
    run_id=f"hivetrain-z",
    batch_size=1,
    target_batch_size=hparams.get("target_batch_size"),
    initial_peers=initial_peers,
    use_ipfs=True,
    use_relay=True,
    use_auto_relay=True,
    verbose=False,
    wait_timeout=30,
    bootstrap_timeout=20,
    matchmaking_time=45.0,
    averaging_timeout=180.0,
    delay_state_averaging=True,
    delay_grad_averaging=True,
    delay_optimizer_step=True,
    offload_optimizer=True,
    reuse_grad_buffers=False,
    # grad_compression=Float16Compression(),
    # state_averaging_compression=Float16Compression(),
    # load_state_compression=NoCompression(),
    scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),#FIXME Am I cylical and infinite?
)

# print my peer id to console
visible_addresses = [
    str(a)
    for a in strategy.dht.get_visible_maddrs()
    if not ipaddress.ip_address(a.values()[0]).is_loopback
]

my_ids = []
pattern = r"(/p2p/.*)"
for peer in list(visible_addresses):
    match = re.search(pattern, peer)
    if match:
        my_ids.append(match.group(1))

for peer in list(set(my_ids)):
    print(
        f"PEER-ID: {peer}"
    )

# define training params
train_params = dict(
    accelerator="auto",
    strategy=strategy,
    devices="auto",
    max_steps=hparams.get("num_steps", 10000) * hparams["target_batch_size"],
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=1,
    precision="32-true",
    accumulate_grad_batches=1, # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    callbacks=[]
)

# set weights as trainable
def get_params(model, hparams):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(nd in n for nd in no_decay):
            weight_decay = 0.0
        else:
            weight_decay = hparams["weight_decay"]

        grouped_parameters.append(
            {
                "params": [p],
                "weight_decay": weight_decay,
            }
        )

    return grouped_parameters

# set model parameters as trainable
params = get_params(model, hparams)

# create the optimizer
optimizer = AdamW(
    params,
    lr=hparams.get("learning_rate", 0.001),
    eps=hparams.get("eps", 1e-8),
)

# for logging progress
class MinerProgressBar(ProgressBar):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.last_step = 0
        self.prev_avg_loss = None
        self.smoothing = 0.01

    def on_train_start(self, trainer, lm):
        super().on_train_start(trainer, lm)
        trainer.pbar = tqdm(
            # total=trainer.estimated_stepping_batches,
            total=self.num_steps,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_train_end(self, trainer, lm):
        trainer.pbar.close()

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        step = int(trainer.callback_metrics.get("step", -1))
        if step == -1:
            return

        current_loss = float(trainer.callback_metrics["train_loss"])

        current_epoch = trainer.current_epoch
        # if lm.train_len > 0:
        #     current_epoch += batch_idx / lm.train_len

        avg_loss = 0
        if not isnan(current_loss):
            avg_loss = self.average_loss(
                current_loss, self.prev_avg_loss, self.smoothing
            )
            self.prev_avg_loss = avg_loss

        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )  # e.g. 4015976448
        mem_gib = mem_bytes / (1024.0**3)  # e.g. 3.74

        memory = psutil.virtual_memory()

        bar = f"Loss: {avg_loss:.3f}"

        if current_epoch > 0:
            bar += f", Epoch: {epoch_string}"

        if hasattr(trainer.strategy, "num_peers"):
            bar += f", Peers: {trainer.strategy.num_peers}"

        if trainer.pbar.n != step:
            trainer.pbar.update(step - trainer.pbar.n)

        # this is a dumb hack to make TQDM print in Docker
        if random.random() < 0.01:
            print()

        trainer.pbar.set_description(bar)

    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss

class ValidationCommunicator(Callback):
    """Periodically save the model during training."""

    def __init__(
        self,
        wallet,
        subtensor,
        metagraph,
        sync_interval = 600
    ):
        super().__init__()
        self.step = 0
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.sync_interval = sync_interval
        self.last_sync_time = 0
        self.validator_urls = []

    def get_validator_uids_and_addresses(self,
    metagraph: "bt.metagraph.Metagraph", vpermit_tao_limit: int = 2000
    ) -> Tuple[List[dict], List[str]]:
    """
    Check availability of all UIDs in a given subnet, returning their IP, port numbers, and hotkeys 
    if they are serving and have at least vpermit_tao_limit stake, along with a list of strings 
    formatted as 'ip:port' for each validator.

    Args:
        metagraph (bt.metagraph.Metagraph): Metagraph object.
        vpermit_tao_limit (int): Validator permit tao limit.

    Returns:
        Tuple[List[dict], List[str]]: A tuple where the first element is a list of dicts with details 
                                      of available UIDs, including their IP, port, and hotkeys, and the 
                                      second element is a list of strings formatted as 'ip:port'.
    """
    available_uid_details = []
    validator_addresses = []  # List to hold 'ip:port' strings
    for uid in range(len(self.metagraph.S)):
        if self.metagraph.S[uid] >= vpermit_tao_limit:
            ip = self.metagraph.axons[uid].ip
            port = self.metagraph.axons[uid].port
            details = {
                "uid": uid,
                "ip": ip,
                "port": port,
                "hotkey": self.metagraph.hotkeys[uid]
            }
            available_uid_details.append(details)
            validator_addresses.append(f"{ip}:{port}")  # Format and add 'ip:port' to the list

    return available_uid_details, validator_addresses
    
    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        self.step = int(trainer.callback_metrics.get("step", 0))

        if self.step % 100:
            if self.should_sync_metagraph():
                self.resync_metagraph()
                _, self.validator_urls = self.get_validator_uids_and_addresses()
            message, signature, public_address = self.create_signed_message(timestamp)

            for url in self.validator_urls:
                try:
                    requests.post(f"http://{url}/validate_metrics", json={"rank": rank, "checksum": checksum, "metrics": metrics})
                except:
                    pass#FIXME log sth
            requests.post()

    def create_signed_message(self, message):
        """Sign a message and return the signature."""
        signature = self.wallet.hotkey.sign(message).hex()  # Convert bytes to hex string for easy transmission
        public_address = wallet.hotkey.ss58_address
        return message, signature, public_address

    def send_metrics(metrics, rank, validator_urls):
        timestamp = time.time()
        message, signature, public_address = create_signed_message(timestamp)
        data = {'message': message, 'signature': signature, 'public_address': public_address, "metrics": metrics, "rank": rank}
        # Ensure metrics is a dictionary
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be provided as a dictionary.")
        # Ensure validator_urls is a list
        if not isinstance(validator_urls, list):
            raise ValueError("validator_urls must be provided as a list.")

    def resync_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)

    def should_sync_metagraph(self):
            """
            Check if enough epoch blocks have elapsed since the last checkpoint to sync.
            """
            return (
                time.time()-self.last_sync_time
            ) > self.sync_interval
            #return (
            #    self.block - self.metagraph.last_update[self.uid]
            #) > self.config.neuron.epoch_length

    def check_registered(self):
            # --- Check for registration.
            if not self.subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address,
            ):
                bt.logging.error(
                    f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                    f" Please register the hotkey using `btcli subnets register` before trying again"
                )
                exit()


train_params["callbacks"].append(MinerProgressBar(hparams.get("num_steps")))
train_params["callbacks"].(ValidationCommunicator())
# Wrap the model in a pytorch-lightning module
train_model = MinerTrainer(
    model,
    optimizer,
    tokenizer,
    hparams
)

# fit the trainer and run
model.train()
trainer = Trainer(**train_params)
trainer.fit(
    train_model,
    dataset
)