import argparse
import ipaddress
import logging
import os
import random
import re
import sys
from functools import partial
from math import isnan

import numpy as np
import torch
from datasets import load_dataset
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.trainer import Trainer
from lightning_hivemind.strategy import HivemindStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

# capture arguments passed to this python script
parser = argparse.ArgumentParser(
    description="Get configs from arguments to this script."
)

parser.add_argument(
    "--initial_peers",
    action="append",
    help="Add a peer. Can be used multiple times to pass multiple peers.",
    nargs="*",
    default=[],
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="The largest batch size able to fit on your GPU.",
    default=1,
    const=1,
    nargs="?",
)

args = parser.parse_args()


def flatten_list(nested_list):
    """Flatten a nested list."""
    if nested_list and isinstance(nested_list[0], list):
        # Assumes only one level of nesting
        return [item for sublist in nested_list for item in sublist]
    return nested_list


# set some basic configuration values
initial_peers = flatten_list(args.initial_peers)

batch_size = args.batch_size
block_size = 1024
num_steps = 100_000
target_batch_size = 256

dataset_config = {
    "dataset": "tiiuae/falcon-refinedweb",
    "key": "content",
    "split": "train",
    "block_size": block_size,
}

# initialized and load the model
config = AutoConfig.from_pretrained(
    "gpt2",
    n_embd=block_size,
    n_ctx=block_size,
    n_layer=8,
    n_head=8,
    n_positions=block_size,
    n_inner=block_size * 4,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    summary_first_dropout=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.05,
    summary_type="cls_index",
    summary_proj_to_labels=True,
    summary_use_proj=True,
    torch_dtype=torch.bfloat16,
)

print(config)

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


# create a datamodule to wrap our remote datasets
class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.train_data = StreamingDataset(self.tokenizer, config)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
        )


# create an iterable dataset, which loops over the streaming data
class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.dataset = load_dataset(
            self.config.get("dataset", "tiiuae/falcon-refinedweb"),
            split=self.config.get("split", "train"),
            streaming=True,
            cache_dir="/tmp/pile",
        )

    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=10000,
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

    def __init__(self, model, optimizer, hparams):
        super(MinerTrainer, self).__init__()

        self.model, self.optimizer = (model, optimizer)
        self.automatic_optimization = True
        self.save_hyperparameters(hparams)

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0]
        self.log(
            "train_loss", float(loss), on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def on_train_batch_end(self, trainer, outputs, idx):
        self.log(
            "step",
            int(self.trainer.strategy.optimizers[0].local_epoch),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return [self.optimizer]


# define the model hyperparameters
hparams = dict(
    learning_rate=0.001,
    weight_decay=0.1,
    eps=1e-8,
    warmup_steps=10,
    batch_size=batch_size,
    num_steps=num_steps,
    block_size=block_size,
)

# define the hivemind strategy
strategy = HivemindStrategy(
    run_id=f"hiveminer",
    batch_size=batch_size,
    target_batch_size=target_batch_size,
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
    scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),
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
    print(f"PEER-ID: {peer}")

# define training params
train_params = dict(
    accelerator="auto",
    strategy=strategy,
    devices="auto",
    max_steps=num_steps * target_batch_size,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=1,
    precision="32-true",
    accumulate_grad_batches=1,  # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    enable_progress_bar=False,
    callbacks=[],
)


# set weights as trainable
def set_trainable_parameters(model, hparams):
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
params = set_trainable_parameters(model, hparams)

# create the optimizer
optimizer = AdamW(
    params,
    lr=hparams.get("learning_rate", 0.001),
    eps=hparams.get("eps", 1e-8),
)


# for logging progress
class MinerConsoleLogging(Callback):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.num_peers = 0
        self.previous_step = None
        self.prev_avg_loss = None

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)
        step = int(trainer.callback_metrics.get("step", -1))
        if step == -1:
            return

        current_loss = float(trainer.callback_metrics["train_loss"])

        avg_loss = 0
        if not isnan(current_loss):
            avg_loss = self.average_loss(current_loss, self.prev_avg_loss)
            self.prev_avg_loss = avg_loss

        output = f"Global Step: {str(step)}, Local Loss: {avg_loss:.3f}"

        if hasattr(trainer.strategy, "num_peers"):
            output += f", Peers: {trainer.strategy.num_peers}"

        if step != self.previous_step or self.num_peers != trainer.strategy.num_peers:
            print(output)
            self.previous_step = step
            self.num_peers = trainer.strategy.num_peers

    def average_loss(self, current_loss, prev_avg_loss, smoothing=0.01):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss


train_params["callbacks"].append(MinerConsoleLogging(hparams.get("num_steps")))

# Wrap the model in a pytorch-lightning module
train_model = MinerTrainer(model, optimizer, hparams)

# fit the trainer and run
model.train()
trainer = Trainer(**train_params)
trainer.fit(train_model, dataset)
