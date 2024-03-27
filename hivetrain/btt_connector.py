import bittensor as bt
import copy
import math
import numpy as np
import bittensor
import torch
import time
from typing import List, Tuple
import bittensor.utils.networking as net
import threading
import logging
from . import __spec_version__

logger = logging.getLogger('waitress')
logger.setLevel(logging.DEBUG)


def initialize_bittensor_objects():
    global wallet, subtensor, metagraph, config
    base_config = copy.deepcopy(config)
    # check_config(base_config)

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
    metagraph = BittensorNetwork.subtensor.metagraph(BittensorNetwork.config.netuid)
    print("Metagraph resynchronization complete.")

def should_sync_metagraph(last_sync_time,sync_interval):
    current_time = time.time()
    return (current_time - last_sync_time) > sync_interval

def sync(last_sync_time, sync_interval, config):
    if should_sync_metagraph(last_sync_time,sync_interval):
        # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
        # This method would need to be defined or adapted from the BaseNeuron implementation.
        try:
            resync_metagraph()
            last_sync_time = time.time()
        except Exception as e:
            logger.warn(f"Failed to resync metagraph: {e}")
        return last_sync_time
    else:
        return last_sync_time



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

    logger.info("serving ip to chain...")
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
            logger.info(
                f"Served Axon {axon} on network: {BittensorNetwork.config.subtensor.chain_endpoint} with netuid: {BittensorNetwork.config.netuid}"
            )
        except Exception as e:
            logger.error(f"Failed to serve Axon with exception: {e}")
            pass

    except Exception as e:
        logger.error(
            f"Failed to create Axon initialize with exception: {e}"
        )
        pass
    return axon



class BittensorNetwork:
    _instance = None
    _lock = threading.Lock()  # Singleton lock
    _weights_lock = threading.Lock()  # Lock for set_weights
    _anomaly_lock = threading.Lock()  # Lock for detect_metric_anomaly
    _config_lock = threading.Lock()  # Lock for modifying config
    _rate_limit_lock = threading.Lock()
    metrics_data = {}
    model_checksums = {}
    request_counts = {}  # Track request counts
    blacklisted_addresses = {}  # Track blacklisted addresses

    


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BittensorNetwork, cls).__new__(cls)
                cls.wallet = None
                cls.subtensor = None
                cls.metagraph = None
                cls.config = None
        return cls._instance

    @classmethod
    def initialize(cls, config):
        with cls._lock:
                cls.wallet = bt.wallet(config=config)
                cls.subtensor = bt.subtensor(config=config)
                cls.metagraph = cls.subtensor.metagraph(config.netuid)
                cls.config = config
                if not cls.subtensor.is_hotkey_registered(netuid=config.netuid, hotkey_ss58=cls.wallet.hotkey.ss58_address):
                    print(f"Wallet: {config.wallet} is not registered on netuid {config.netuid}. Please register the hotkey before trying again")
                    exit()
                cls.uid = cls.metagraph.hotkeys.index(
                    cls.wallet.hotkey.ss58_address
                )
                cls.device="cpu"
                cls.base_scores = torch.zeros(
                    cls.metagraph.n, dtype=torch.float32, device=cls.device
                )
            # Additional initialization logic here

    @classmethod
    def set_weights(cls, scores):
        try:
            #chain_weights = torch.zeros(cls.subtensor.subnetwork_n(netuid=cls.metagraph.netuid))
            uids = []
            for uid, public_address in enumerate(cls.metagraph.hotkeys):
                try:
                    alpha = 0.333333 # T=5 (2/(5+1))
                    cls.base_scores[uid] = alpha * scores.get(public_address, 0) + (1 - alpha) * cls.base_scores[uid].to(cls.device)                    
                    uids.append(uid)
                except KeyError:
                    continue
            uids = torch.tensor(uids)
            logger.info(f"raw_weights {cls.base_scores}")
            logger.info(f"raw_weight_uids {uids}")
            # Process the raw weights to final_weights via subtensor limitations.
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids.to("cpu"),
                weights=cls.base_scores.to("cpu"),
                netuid=cls.config.netuid,
                subtensor=cls.subtensor,
                metagraph=cls.metagraph,
            )
            logger.info(f"processed_weights {processed_weights}")
            logger.info(f"processed_weight_uids {processed_weight_uids}")

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            logger.info("Sending weights to subtensor")
            result = cls.subtensor.set_weights(
                wallet=cls.wallet,
                netuid=cls.metagraph.netuid,
                uids=uint_uids, 
                weights=uint_weights,
                wait_for_inclusion=False,
                version_key=__spec_version__
            )
        except Exception as e:
            logger.info(f"Error setting weights: {e}")

    @classmethod
    def should_set_weights(cls) -> bool:
        with cls._lock:  # Assuming last_update modification is protected elsewhere with the same lock
            return (cls.subtensor.get_current_block() - cls.metagraph.last_update[cls.uid]) > cls.config.neuron.epoch_length

    @classmethod
    def detect_metric_anomaly(cls, metric="loss", OUTLIER_THRESHOLD=2, MEDIAN_ABSOLUTE_DEVIATION=True):
        from scipy.stats import median_abs_deviation
        with cls._anomaly_lock:
            if not cls.metrics_data:
                return {}

            logger.info(f"Metrics Data: {cls.metrics_data}")
            aggregated_metrics = {}
            for public_address, data in cls.metrics_data.items():
                if metric in data:
                    if public_address in aggregated_metrics:#FIXME no need for an if condition
                        aggregated_metrics[public_address].append(data[metric])
                    else:
                        aggregated_metrics[public_address] = [data[metric]]

            if MEDIAN_ABSOLUTE_DEVIATION:
                # Use Median Absolute Deviation for outlier detection
                values = [np.median(vals) for vals in aggregated_metrics.values()]
                median = np.median(values)
                deviation = median_abs_deviation(values, scale='normal')
                is_outlier = {}
                for addr, vals in aggregated_metrics.items():
                    try:
                        is_outlier[addr] = (abs(np.median(vals) - median) / deviation) > OUTLIER_THRESHOLD
                    except:
                        is_outlier[addr] = True
                    
            else:
                # Use Mean and Standard Deviation for outlier detection
                average_metrics = {addr: np.nanmean(vals) for addr, vals in aggregated_metrics.items()}
                losses = np.array(list(average_metrics.values()))
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                is_outlier = {addr: abs(avg_loss - mean_loss) / std_loss > OUTLIER_THRESHOLD 
                            for addr, avg_loss in average_metrics.items()}

            scores = {public_address: 0 if is_outlier.get(public_address, False) else 1 for public_address in aggregated_metrics}
            logger.info(f"Scores calculated: {scores}")
            return scores


    @classmethod
    def run_evaluation(cls):
        #global model_checksums, metrics_data
        logger.info("Evaluating miners")
        # checksum_frequencies = {}
        # for public_address, checksum in cls.model_checksums.items():
        #     checksum_frequencies[public_address] = checksum_frequencies.get(public_address, 0) + 1
        
        # model_scores = {}
        # try:
        #     most_common_checksum = max(checksum_frequencies, key=checksum_frequencies.get)
        #     model_scores = {public_address: (1 if checksum == most_common_checksum else 0) for public_address, checksum in cls.model_checksums.items()}
        #     logger.info("Model scores based on checksum consensus:", model_scores)

        # except ValueError:
        #     pass        
        
        with cls._weights_lock:
            if BittensorNetwork.should_set_weights():
                scores = BittensorNetwork.detect_metric_anomaly()
                BittensorNetwork.set_weights(scores)

                cls.model_checksums.clear()
                cls.metrics_data.clear()
    
    @classmethod
    def rate_limiter(cls, public_address, n=10, t=60):
        """
        Check if a public_address has exceeded n requests in t seconds.
        If exceeded, add to blacklist.
        """
        with cls._rate_limit_lock:
            current_time = time.time()
            if public_address in cls.blacklisted_addresses:
                # Check if the blacklist period is over
                if current_time - cls.blacklisted_addresses[public_address] > t:
                    del cls.blacklisted_addresses[public_address]
                else:
                    return False  # Still blacklisted

            request_times = cls.request_counts.get(public_address, [])
            # Filter out requests outside of the time window
            request_times = [rt for rt in request_times if current_time - rt <= t]
            request_times.append(current_time)
            cls.request_counts[public_address] = request_times

            if len(request_times) > n:
                logger.info(f"Blacklisted {public_address} for making {len(request_times)} in {t} seconds")
                cls.blacklisted_addresses[public_address] = current_time
                return False  # Too many requests, added to blacklist

            return True  # Request allowed
