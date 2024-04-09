#Thanks SN9

import multiprocessing
import functools
import bittensor as bt
import os
import lzma
import base64
import multiprocessing
from typing import Optional, Any
from bittensor.btlogging import logging


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)

def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


class ChainMultiAddressStore:
    """Chain based implementation for storing and retrieving multiaddresses."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        wallet: Optional[bt.wallet] = None,
        
    ):
        self.subtensor = subtensor
        self.wallet = wallet
        self.subnet_uid = subnet_uid

    def store_hf_repo(self, hf_repo: str):
        """Stores compressed multiaddress on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # Compress the multiaddress

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            self.subtensor.commit,
            self.wallet,
            self.subnet_uid,
            multiaddress,
        )
        run_in_subprocess(partial, 60)

    def retreive_hf_repo(self, hotkey: str) -> Optional[str]:
        """Retrieves and decompresses multiaddress on this subnet for specific hotkey"""
        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            bt.extrinsics.serving.get_metadata, self.subtensor, self.subnet_uid, hotkey
        )

        try:
            metadata = run_in_subprocess(partial, 60)
        except:
            metadata = None
            logging.warning(f"Failed to retreive multiaddress for: {hotkey}")
            

        if not metadata:
            return None

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        multiaddress = bytes.fromhex(hex_data).decode()

        try:
            return multiaddress
        except:
            # If the data format is not correct or decompression fails, return None.
            bt.logging.trace(
                f"Failed to parse the data on the chain for hotkey {hotkey}."
            )
            return None

# Synchronous test cases for ChainMultiAddressStore


class LocalAddressStore:
    """Simulated local storage for storing and retrieving multiaddresses."""
    
    def __init__(self, subtensor:None = None,
        subnet_uid: None = None,
        wallet: Optional[str] = None):
        # Using a simple dictionary to simulate storage
        self.storage = {}
        self.wallet = wallet

    def store_hf_repo(self, hf_repo: str):
        """Stores the Hugging Face repository link for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the storage.")
        
        # Store the HF repository link in the simulated local storage
        self.storage[self.wallet.hotkey.ss58_address] = hf_repo
        print(f"Stored {hf_repo} for wallet {self.wallet}")

    def retrieve_hf_repo(self, hotkey: str) -> Optional[str]:
        """Retrieves the Hugging Face repository link for a specific wallet."""
        # Retrieve the HF repository link from the simulated local storage
        hf_repo = self.storage.get(hotkey)
        if hf_repo:
            print(f"Retrieved {hf_repo} for wallet {wallet}")
            return hf_repo
        else:
            logging.warning(f"Failed to retrieve repository for: {wallet}")
            return None

def test_store_multiaddress():
    """Verifies that the ChainMultiAddressStore can store data on the chain."""
    multiaddress = "/ip4/198.51.100.0/tcp/4242/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N"

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    address_store = ChainMultiAddressStore(subtensor, wallet, net_uid)

    # Store the multiaddress on chain.
    address_store.store_multiaddress(hotkey, multiaddress)

    print(f"Finished storing multiaddress for {hotkey} on the chain.")


def test_retrieve_multiaddress():
    """Verifies that the ChainMultiAddressStore can retrieve data from the chain."""
    expected_multiaddress = "/ip4/198.51.100.0/tcp/4242/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N"

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    subtensor = bt.subtensor()

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    hotkey = os.getenv("TEST_HOTKEY")

    address_store = ChainMultiAddressStore(subtensor, None, net_uid)

    # Retrieve the multiaddress from the chain.
    retrieved_multiaddress = address_store.retrieve_multiaddress(hotkey)

    print(f"Retrieved multiaddress matches expected: {expected_multiaddress == retrieved_multiaddress}")


def test_roundtrip_multiaddress():
    """Verifies that the ChainMultiAddressStore can roundtrip data on the chain."""
    multiaddress = "/ip4/198.51.100.0/tcp/4242/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N"

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    address_store = ChainMultiAddressStore(subtensor, wallet, net_uid)

    # Store the multiaddress on chain.
    address_store.store_multiaddress(hotkey, multiaddress)

    # Retrieve the multiaddress from the chain.
    retrieved_multiaddress = address_store.retrieve_multiaddress(hotkey)

    print(f"Expecting matching multiaddress: {multiaddress == retrieved_multiaddress}")



if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    # asyncio.run(test_roundtrip_model_metadata())
    # asyncio.run(test_store_model_metadata())
    test_retrieve_model_metadata()

