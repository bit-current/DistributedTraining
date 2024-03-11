import bittensor as bt
import copy
import math
import time
from typing import List, Tuple
import bittensor.utils.networking as net



def initialize_bittensor_objects():
    global wallet, subtensor, metagraph, config
    base_config = copy.deepcopy(config)
    check_config(base_config)

    if base_config.mock:
        wallet = bt.MockWallet(config=base_config)
        subtensor = MockSubtensor(base_config.netuid, wallet=wallet)
        metagraph = MockMetagraph(base_config.netuid, subtensor=subtensor)
    else:
        wallet = bt.wallet(config=base_config)
        subtensor = bt.subtensor(config=base_config)
        metagraph = subtensor.metagraph(base_config.netuid)


def check_registered(netuid):
    
    if not BittensorNetwork.subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=BittensorNetwork.wallet.hotkey.ss58_address):
        print(f"Wallet: {wallet} is not registered on netuid {netuid}. Please register the hotkey before trying again")
        exit()

def resync_metagraph():
    global metagraph, config, subtensor
    # Fetch the latest state of the metagraph from the Bittensor network
    print("Resynchronizing metagraph...")
        # Update the metagraph with the latest information from the network
    metagraph = BittensorNetwork.subtensor.metagraph(config.netuid)
    print("Metagraph resynchronization complete.")

def should_sync_metagraph(last_sync_time,sync_interval):
    current_time = time.time()
    return (current_time - last_sync_time) > sync_interval

def sync(last_sync_time, sync_interval):
    if should_sync_metagraph(last_sync_time,sync_interval):
        # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
        # This method would need to be defined or adapted from the BaseNeuron implementation.
        resync_metagraph()
        last_sync_time = time.time()
        return last_sync_time
    else:
        return last_sync_time

def get_validator_uids_and_addresses(
    metagraph: "bt.metagraph.Metagraph", vpermit_tao_limit: int
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
    for uid in range(len(metagraph.S)):
        if metagraph.S[uid] >= vpermit_tao_limit:
            ip = metagraph.axons[uid].ip
            port = metagraph.axons[uid].port
            details = {
                "uid": uid,
                "ip": ip,
                "port": port,
                "hotkey": metagraph.hotkeys[uid]
            }
            available_uid_details.append(details)
            validator_addresses.append(f"{ip}:{port}")  # Format and add 'ip:port' to the list

    return available_uid_details, validator_addresses

# def serve_on_subtensor(external_ip, external_port, netuid, max_retries=5, wait_for_inclusion=True, wait_for_finalization=False):
#     retry_count = 0
#     check_registered(netuid)
#     while retry_count < max_retries:
#         try:
#             breakpoint()
#             serve_success = BittensorNetwork.subtensor.serve(
#                 wallet=BittensorNetwork.wallet,
#                 ip=external_ip,
#                 port=external_port,
#                 netuid=netuid,
#                 protocol=4,
#                 wait_for_inclusion=wait_for_inclusion,
#                 wait_for_finalization=wait_for_finalization,
#                 prompt=False,
#             )
#             if serve_success:
#                 print(f"Serving on IP: {external_ip}, Port: {external_port}")
#                 break
#             else:
#                 print("Failed to serve on Subtensor network. Retrying...")
#         except Exception as e:
#             print(f"Error serving on Subtensor network: {e}")
        
#         retry_count += 1
#         sleep_time = math.pow(2, retry_count)  # Exponential backoff
#         print(f"Retry {retry_count}/{max_retries}. Retrying in {sleep_time} seconds.")
#         time.sleep(sleep_time)

#     if retry_count == max_retries:
#         print("Max retries reached. Failed to serve on Subtensor network.")

def serve_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    ip: str,
    port: int,
    protocol: int,
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
    prompt: bool = False,
) -> bool:
    r"""Subscribes a Bittensor endpoint to the subtensor chain.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        ip (str):
            Endpoint host port i.e., ``192.122.31.4``.
        port (int):
            Endpoint port number i.e., ``9221``.
        protocol (int):
            An ``int`` representation of the protocol.
        netuid (int):
            The network uid to serve on.
        placeholder1 (int):
            A placeholder for future use.
        placeholder2 (int):
            A placeholder for future use.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # Decrypt hotkey
    wallet.hotkey
    params: "bittensor.AxonServeCallParams" = {
        "version": bittensor.__version_as_int__,
        "ip": net.ip_to_int(ip),
        "port": port,
        "ip_type": net.ip_version(ip),
        "netuid": netuid,
        "hotkey": wallet.hotkey.ss58_address,
        "coldkey": wallet.coldkeypub.ss58_address,
        "protocol": protocol,
        "placeholder1": placeholder1,
        "placeholder2": placeholder2,
    }
    bittensor.logging.debug("Checking axon ...")
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )
    neuron_up_to_date = not neuron.is_null and params == {
        "version": neuron.axon_info.version,
        "ip": net.ip_to_int(neuron.axon_info.ip),
        "port": neuron.axon_info.port,
        "ip_type": neuron.axon_info.ip_type,
        "netuid": neuron.netuid,
        "hotkey": neuron.hotkey,
        "coldkey": neuron.coldkey,
        "protocol": neuron.axon_info.protocol,
        "placeholder1": neuron.axon_info.placeholder1,
        "placeholder2": neuron.axon_info.placeholder2,
    }
    output = params.copy()
    output["coldkey"] = wallet.coldkeypub.ss58_address
    output["hotkey"] = wallet.hotkey.ss58_address
    if neuron_up_to_date:
        bittensor.logging.debug(
            f"Axon already served on: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) "
        )
        return True

    if prompt:
        output = params.copy()
        output["coldkey"] = wallet.coldkeypub.ss58_address
        output["hotkey"] = wallet.hotkey.ss58_address
        if not Confirm.ask(
            "Do you want to serve axon:\n  [bold white]{}[/bold white]".format(
                json.dumps(output, indent=4, sort_keys=True)
            )
        ):
            return False

    bittensor.logging.debug(
        f"Serving axon with: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) -> {subtensor.network}:{netuid}"
    )
    params["ip"] = net.int_to_ip(params["ip"])
    success, error_message = subtensor._do_serve_axon(
        wallet=wallet,
        call_params=params,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
    )

    if wait_for_inclusion or wait_for_finalization:
        if success == True:
            bittensor.logging.debug(
                f"Axon served with: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) on {subtensor.network}:{netuid} "
            )
            return True
        else:
            bittensor.logging.debug(
                f"Axon failed to served with error: {error_message} "
            )
            return False
    else:
        return True

def serve_axon(netuid,host_address,external_address, host_port, external_port):
    """Serve axon to enable external connections."""

    bt.logging.info("serving ip to chain...")
    try:
        axon = bt.axon(
            config=BittensorNetwork.config,
            wallet=BittensorNetwork.wallet,
            # port=host_port,
            # ip=host_address,
            # external_ip=external_address,
            # external_port=external_port
        )
        axon.external_ip = external_address
        axon.external_port = external_port
        try:
            # BittensorNetwork.subtensor.serve_axon(
            #     netuid=netuid,
            #     axon=axon,
            # )
            serve_success = BittensorNetwork.subtensor.serve(
                wallet=BittensorNetwork.wallet,
                ip=external_address,
                port=external_port,
                netuid=netuid,
                protocol=4,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                prompt=False,
            )
            if serve_success:
                print("success")
            else:
                print("ARGH")
            bt.logging.info(
                f"Served Axon {axon} on network: {BittensorNetwork.config.subtensor.chain_endpoint} with netuid: {BittensorNetwork.config.netuid}"
            )
        except Exception as e:
            bt.logging.error(f"Failed to serve Axon with exception: {e}")
            pass

    except Exception as e:
        bt.logging.error(
            f"Failed to create Axon initialize with exception: {e}"
        )
        pass
    return axon


class BittensorNetwork:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BittensorNetwork, cls).__new__(cls)
            cls.wallet = None
            cls.subtensor = None
            cls.metagraph = None
        return cls._instance

    @classmethod
    def initialize(cls, config):
        if config.mock:
            cls.wallet = bt.MockWallet(config=config)
            cls.subtensor = bt.subtensor.mock(config=config)
            cls.metagraph = cls.subtensor.metagraph()
            cls.config = config
        else:
            cls.wallet = bt.wallet(config=config)
            cls.subtensor = bt.subtensor(config=config)
            cls.metagraph = cls.subtensor.metagraph(config.netuid)
            cls.config = config

        # Additional initialization logic here
