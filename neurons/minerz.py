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

from lightning_hivemind.strategy import HivemindStrategy
from functools import partial

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from datasets import load_dataset

# initialized and load the model
config = AutoConfig.from_pretrained(
    "gpt2",
    n_emb=256,
    n_layer=3,
    n_head=3,
    n_inner=3,
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

# create a datamodule to wrap our remote datasets
class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.train_data = None
        # self.setup(self, self.config, stage=None)
        self.train_data = StreamingDataset(
            self.tokenizer, config
        )

    # def setup(self, config, stage=None):


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

        block_size = 512

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
dataset = StreamingDataModule(tokenizer, {
    "dataset": "tiiuae/falcon-refinedweb",
    "key": "content",
    "split": "train",
})

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
        self.save_hyperparameters(hparams)

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0]
        return loss

    def on_train_batch_end(self, trainer, lm, outputs):
        schedule = self.lr_schedulers()
        step = self.global_step

        if hasattr(schedule, "current_step"):
            step = schedule.current_step
        elif hasattr(self.trainer.strategy.optimizers[0], "local_epoch"):
            step = self.trainer.strategy.optimizers[0].local_epoch

        if hasattr(schedule, "step"):
            schedule.step()

    def configure_optimizers(self):
        "Create optimizer and scheduler"

        # if self.scheduler:
        #     return [self.optimizer], [self.scheduler()]
        return [self.optimizer]

# define the model hyperparameters
hparams = dict(
    # optimizer="AdamW",
    # scheduler=scheduler,
    learning_rate=0.001,
    weight_decay=0.1,
    eps=1e-8,
    warmup_steps=0,
    batch_size=1,
    num_steps=10000,
    block_size=512
)

# define the hivemind strategy
initial_peers = hparams.get("initial_peers", [])
strategy = HivemindStrategy(
    run_id=f"hivetrain-z",
    batch_size=1,
    target_batch_size=256,
    initial_peers=initial_peers,
    use_ipfs=True,
    use_relay=True,
    use_auto_relay=True,
    verbose=True,
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

# define training params
train_params = dict(
    accelerator="auto",
    strategy=strategy,
    devices="auto",
    max_steps=10000,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=1,
    precision="32-true",
    accumulate_grad_batches=1, # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    # callbacks=callbacks,
    # logger=loggers if loggers else False,
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