import ipaddress
import logging
import os
import random
import re
import sys
import time
import bittensor as bt
from bittensor import metagraph

import requests

from hivetrain.btt_connector import (
    BittensorNetwork,
)
from hivetrain.config import Configurator
from hivetrain import __spec_version__
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
logger = logging.getLogger("lightning.pytorch")

args = Configurator.combine_configs()

class ValidationCommunicator:
    """Periodically send dummy requests to validators."""

    def __init__(self, args, sync_interval=600):
        BittensorNetwork.initialize(args)

        self.wallet = BittensorNetwork.wallet
        self.subtensor = BittensorNetwork.subtensor
        self.metagraph = BittensorNetwork.metagraph
        self.sync_interval = sync_interval
        self.last_sync_time = 0
        self.validator_urls = ["127.0.0.1:8888"]

    def start(self):
        while True:
            current_time = int(time.time())
            if self.should_sync_metagraph():
                self.resync_metagraph()
            timestamp = str(current_time)
            message, signature, public_address = self.create_signed_message(timestamp)

            for url in self.validator_urls:
                try:
                    response = requests.post(
                        f"http://{url}/validate_metrics",
                        json={"metrics": {"loss": random.random()},
                              "message": message, "signature": signature, "public_address": public_address, "miner_version": __spec_version__},
                        timeout=3
                    )
                    if response.status_code == 200:
                        logger.info(f"Dummy metrics reported successfully to validator {url}")
                    else:
                        logger.warn(f"Error @ validator {url} --- Error: {response.json()['error']}")
                except Exception as e:
                    logger.warn(f"Failed to confirm reception at {url}: {str(e)}")

            time.sleep(60)  # Sleep for 60 seconds before sending the next request

    def create_signed_message(self, message):
        signature = self.wallet.hotkey.sign(
            message
        ).hex()
        public_address = self.wallet.hotkey.ss58_address
        return message, signature, public_address

    def resync_metagraph(self):
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            logger.info("Syncing Metagraph Successful")
        except Exception as e:
            logger.warning(f"Failed to sync metagraph: {e}")

    def should_sync_metagraph(self):
        return (time.time() - self.last_sync_time) > self.sync_interval

if __name__ == "__main__":
    communicator = ValidationCommunicator(args)
    communicator.start()