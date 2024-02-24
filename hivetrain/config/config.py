import os
import torch
import argparse
import bittensor as bt
from loguru import logger
from argparse import ArgumentParser
import bittensor as bt
from .hivetrain_config import add_meta_miner_args, add_orchestrator_args, add_torch_miner_args #s, add_validator_args
from .base_subnet_config import add_neuron_args, add_validator_args, add_miner_args


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )

class Configurator:
    @staticmethod
    def combine_configs():
        parser = ArgumentParser(description="Unified Configuration for Bittensor")
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)

        add_torch_miner_args(parser)
        add_meta_miner_args(parser)
        add_orchestrator_args(parser)
        add_neuron_args(parser)
        add_miner_args(parser)
        add_validator_args(parser)
        args = parser.parse_args()
        return bt.config(parser)