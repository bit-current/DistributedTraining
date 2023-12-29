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


#from ast import If
import hivemind
import time

import bittensor as bt

import torch
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial
import template
from template.utils.config import get_config
from template.utils.misc import AsyncDendritePool, load_wandb
#from template.utils.uids import get_random_uids
from template.validator.validator_core import DatasetStateSingelton, ModelSingleton, upload_checkpoint, DHTManager
#from template.validator import forward
#from template.base.validator import BaseValidatorNeuron
from template.utils.uids import check_uid_availability

from typing import List
import traceback
import random 
import wandb

def get_rewards(
    state_averager,
    dataset_common_state,
    config,
    uids,
    dataset,
    encode,
    model,
    device = "cuda"
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    try:
        state_averager.load_state_from_peers()
    except Exception as e:
        breakpoint()

    global_step = dataset_common_state.get_dht("step")
    if global_step % 100 == 0:
        dataset_indices_list_test = (
            dataset_common_state.get_dataset_indices_test(
                config.neuron.batch_size_test
            )
        )

    # Select the correct datapoints
    dataset_sample = dataset.select(dataset_indices_list_test)

    # Encode the dataset
    encoded_dataset = dataset_sample.map(encode, batched=True)

    # Move batch to device
    input_ids = torch.tensor(encoded_dataset["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded_dataset["attention_mask"]).to(device)
    labels = torch.tensor(encoded_dataset["input_ids"]).to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )

    # Backward pass
    loss = outputs.loss

    # Get latest previous loss from DHT
    previous_loss = dataset_common_state.get_dht("loss")
    bt.logging.info(f"Previous loss:    {previous_loss}")
    bt.logging.info(f"Current loss:     {loss}")
    if not config.neuron.dont_wandb_log:
        wandb.log({"loss": loss, "previous_loss": previous_loss})


    # Compute score
    if (previous_loss is None) or ((loss - previous_loss) > 0):
        score = 1
        dataset_common_state.set_dht("loss", float(loss))
    else:
        score = 0

    # Log score, previous and current loss
    bt.logging.info(f"Score:            {score}")

    # Set previous loss to current loss
    previous_loss = loss

    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor([score for _ in uids]).to(device)


def get_random_uids(
    metagraph,
    k: int, exclude:List[int] = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(
            metagraph, uid, config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        uids = torch.tensor(available_uids)
    else:
        uids = torch.tensor(random.sample(available_uids, k))
        
    return uids

def update_scores(scores: torch.FloatTensor, rewards: torch.FloatTensor, uids: List[int],device="cuda"):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = scores.scatter(
            0, torch.tensor(uids).to(device), rewards
        ).to(device)
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = config.neuron.moving_average_alpha
        scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * scores.to(device)
        bt.logging.debug(f"Updated moving avg scores: {scores}")
        return scores

def main(config):
    global blacklisted_hotkeys_set
    global blacklisted_coldkeys_set

    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Optimize the blacklist list TODO add me
    #blacklisted_hotkeys_set = {blacklisted_hotkey for blacklisted_hotkey in config.blacklisted_hotkeys}
    #blacklisted_coldkeys_set = {blacklisted_coldkey for blacklisted_coldkey in config.blacklisted_coldkeys}

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")

    # Initialize alpha
    alpha = 0.9

    # Initialize weights for each miner, store current uids.
    last_uids = metagraph.uids.tolist()
    scores = torch.zeros(len(last_uids), dtype=torch.float32)

    curr_block = subtensor.block
    last_updated_block = curr_block - (curr_block % 100)

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")

    # Init DHT
    #dht = hivemind.DHT(initial_peers=[config.neuron.initial_peers], start=True)
    #dht = DHTManager( initial_peers=[config.neuron.initial_peers], start=True, host_maddrs=[f"/ip4/0.0.0.0/tcp/42175"], client_mode=False)#, ],
    dht = hivemind.DHT(initial_peers=[config.neuron.initial_peers], start=True, 
            host_maddrs=[f"/ip4/0.0.0.0/tcp/42175"], 
            announce_maddrs=["/ip4/24.109.105.158/tcp/42175"],
            client_mode=False)#, ],)
    dataset_dht = dict()#DHTManager(logger_name="DatasetDHT", initial_peers=[config.neuron.initial_peers], start=True, host_maddrs=[f"/ip4/0.0.0.0/tcp/32693"])#host_maddrs=[f"/ip4/0.0.0.0/tcp/{config.neuron.dht_port_2}"],

    # Init Dataset
    dataset = load_dataset(config.neuron.dataset_name, 'wikitext-2-v1', split='train')
    dataset_indices = [i for i in range(0, len(dataset))]
    dataset_common_state = DatasetStateSingelton(dataset_dht , dataset_indices, config.neuron.run_id)

    # Init Loss
    previous_loss = dataset_common_state.get_dht("loss")
    latest_upload = 0
    latest_weight_update = 0
    step = 0

    # Init device
    device = config.neuron.device

    # Init Model
    model = ModelSingleton.get_instance(config.neuron.model_name, config.neuron.device)
    tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Init State Averager
    opt = torch.optim.AdamW(model.parameters(), lr=config.neuron.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda t: 1.0 / max(1, t))

    state_averager = TrainingStateAverager(
        dht=dht, 
        optimizer=opt,
        scheduler=scheduler,
        params=model.parameters(),
        allow_state_sharing=True,
        start=True,
        prefix=f"{config.neuron.run_id}_state_averager", 
        client_mode=False,
        # state_compression=hivemind.Float16Compression(),
        # bandwidth=optimizer_args.bandwidth,
        # client_mode=optimizer_args.client_mode,
        # **asdict(averager_args),
    )

    # Start Main Validation Loop
    bt.logging.info("Starting validator loop.")
        
    # Define encoding function
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

    if not config.neuron.dont_wandb_log:
        wandb = load_wandb(config, wallet)


    step = 0
 

    while True:

        
        #bt.logging.info(f"step({step}) block({block})")

        #TODO change get_random_uids
        miner_uids = get_random_uids(
            metagraph,
            k=config.neuron.sample_size
        )


        datapoints_per_group = config.neuron.target_batch_size

        dataset_indices_list = (
            dataset_common_state.get_dataset_indices(
                groups_count=len(miner_uids),
                items_per_group=datapoints_per_group,
            )
        )  # TODO add repeat on blocked

        if not config.neuron.dont_wandb_log:
            wandb.log({"uids":miner_uids,
                        "dataset_indices":dataset_indices_list})

        # Run multiple forwards concurrently.
        
          # TODO add loss anomaly detection

        # blocking component
        # Adjust the scores based on responses from miners.

        
        step += 1

        try:

            # Sync the subtensor state with the blockchain.
            if step % 5 == 0:

                # Resync our local state with the latest state from the blockchain.
                metagraph = subtensor.metagraph(config.netuid)

                # Sync scores with metagraph
                # Get the current uids of all miners in the network.
                uids = metagraph.uids.tolist()
                # Create new_scores with current metagraph
                
            #TODO add autoupdate + update check 
            #if step % 10 == 0:
                # Check for auto update
                #if config.auto_update == "yes":
                #    compute.util.try_update()
                # Filter axons with stake and ip address.
                # queryable_uids = get_valid_queryable_uids(metagraph, uids)
                # queryable_axons = [metagraph.axons[metagraph.uids.tolist().index(uid)] for uid in queryable_uids]
                # axons_list, uids_list, hotkeys_list = filter_axons(
                #     axons_list=queryable_axons,
                #     uids_list=queryable_uids,
                # )
                
            dataset_dict = dict()
            for uid, uid_dataset in zip(miner_uids, dataset_indices_list):
                #TODO get the hashes of the groups
                # Make miners return the group hashes with their losses as well.
                #UID dict 
                #queries.append(
                #    template.protocol.Train( 
                #        dataset_indices = uid_dataset,
                #        run_id = config.neuron.run_id,
                #        batch_size = config.neuron.batch_size_train  #TODO let miners decide this? Based on their hardware. Then reconcile if needed?
                #    )
                #)
                dataset_dict[uid] = uid_dataset
            
            # TODO Fix miner on the other side
            # Query the miners for benchmarking
            training_synapse = template.protocol.Train( 
                dataset_indices = dataset_dict,
                run_id = config.neuron.run_id,
                batch_size = config.neuron.batch_size_train  #TODO let miners decide this? Based on their hardware. Then reconcile if needed?
            )    
            
            responses = dendrite.query(
                miner_uids,
                training_synapse,
                timeout=120,
            )

            try:
                bt.logging.info(f"Received responses: {[{'Loss':response.loss,'Dataset Indices':(min(response.dataset_indices), max(response.dataset_indices)), 'IP':response.dendrite.ip, 'Port':response.dendrite.port, 'Hotkey':response.dendrite.hotkey} for response in responses[0] if response.dendrite.status_code == 200 ]}")
            except TypeError:
                if responses is None:
                    bt.logging.info("Empty response")
    
            # Calculate response score            

            #TODO add the individual response check

            # Calculate total score

            #TODO get test dataset


            

            new_rewards = get_rewards(
                state_averager,
                dataset_common_state,
                config,
                miner_uids,
                dataset,
                encode,
                model,
                device = "cuda"
            ) 

            bt.logging.info(f"Scored responses: {new_rewards}")
            # Update the scores based on the rewards.
            scores = update_scores(scores, new_rewards, miner_uids)
            # Check if we should exit.
            # Update global and local step
            dataset_common_state.update_step()

            # Periodically update the weights on the Bittensor blockchain.
            current_block = subtensor.block
            if current_block - last_updated_block > 100:
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0) #FIXME bug where it normalizes to miner_ids
                bt.logging.info(f"🏋️ Weight of miners : {weights.tolist()}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid=config.netuid,  # Subnet to set weights on.
                    wallet=wallet,  # Wallet to sign set weights using hotkey.
                    uids=miner_uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    wait_for_inclusion=False,
                )
                last_updated_block = current_block
                if result:
                    bt.logging.success("Successfully set weights.")
                else:
                    bt.logging.error("Failed to set weights.")

            # End the current step and prepare for the next iteration.
            step += 1
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            # time.sleep(bt.__blocktime__)
            time.sleep(10)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            #opt.shutdown()
            dht.shutdown()
            dataset_dht.shutdown()
            exit()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main(config)