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

import time
import typing
from functools import partial

import bittensor as bt
import hivemind

# Bittensor Miner Template:
import template
import torch
from datasets import load_dataset

# import base miner class which takes care of most of the boilerplate
#from template.base.miner import BaseMinerNeuron
from template.utils.misc import load_wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
import wandb
import traceback
from template.utils.config import get_config

# Main takes the config and starts the miner.
def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    #bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 5: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.

        # Init device
    device = config.neuron.device

    # Init DHT and model
    dht = hivemind.DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{config.dht.port}", f"/ip4/0.0.0.0/udp/{config.dht.port}/quic"],
        announce_maddrs=[f"/ip4/38.79.71.1/tcp/{config.dht.port}", f"/ip4/38.79.71.1/udp/{config.dht.port}/quic"],
        initial_peers=[config.neuron.initial_peers], 
        start=True,
        client_mode = False
    )
    model = AutoModelForCausalLM.from_pretrained(config.neuron.model_name)

    # Move the model to the appropriate device
    model = model.to(device)
    # Set up a decentralized optimizer that will average with peers in background
    opt = torch.optim.AdamW(model.parameters(), lr=config.neuron.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda t: 1.0 / max(1, t))

    opt = hivemind.Optimizer(
        dht=dht,                    # use a DHT that is connected with other peers
        run_id=config.neuron.run_id,        # unique identifier of this collaborative run
        scheduler=scheduler,
        batch_size_per_step=config.neuron.batch_size_train,     # each call to opt.step adds this many samples towards the next epoch
        target_batch_size=config.neuron.target_batch_size,    # after peers collectively process this many samples, average weights and begin the next epoch
        optimizer=opt,              # wrap the SGD optimizer defined above
        #use_local_updates=True,     # perform optimizer steps with local gradients, average parameters in background
        matchmaking_time=120.0,       # when averaging parameters, gather peers in background for up to this many seconds
        averaging_timeout=120.0,     # give up on averaging if not successful in this many seconds
        verbose=True               # print logs incessently
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
    # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(config.neuron.dataset_name, 'wikitext-2-v1', split='train')
    
    # Define encoding function
    def encode(self, examples):
        return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

    # The blacklist function decides if a request should be ignored.
    #TODO add synapse type
    def blacklist(synapse: template.protocol.Train) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Train): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        hotkey = synapse.dendrite.hotkey
        synapse_type = type(synapse).__name__

        uid = None
        axon = None
        for _uid, _axon in enumerate(metagraph.axons):
            if _axon.hotkey == hotkey:
                uid = _uid
                axon = _axon
                break

        if uid is None:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
            )
            return (
                True,
                f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
            )

        if config.blacklist.force_validator_permit and (not config.blacklist.allow_non_registered):
            # Check stake if uid is recognize
            tao = metagraph.neurons[uid].stake.tao
            if tao < config.neuron.vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {tao} < {config.neuron.vpermit_tao_limit} from {hotkey}",
                )

        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    #TODO synapse type
    def priority(synapse: template.protocol.Train) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Train): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    # This is the Allocate function, which decides the miner's response to a valid, high-priority request.
    def forward(synapse: template.protocol.Train) -> template.protocol.Train:
        """
        Processes the incoming 'Train' synapse by performing a training run

        Args:
            synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

        Returns:
            template.protocol.Train: The synapse object with the 'loss' field set to models loss.
        """

        bt.logging.info("Loading state from peers")

        while not opt.is_synchronized_with_peers():
            try:
                opt.load_state_from_peers()
            except Exception as e:
                bt.logging.error("Unable to load state from peers.")

        if opt.is_synchronized_with_peers():
            bt.logging.info("Miner synchronized with peers")

        # Select dataset indices to use for optimization step
        dataset = dataset.select(synapse.dataset_indices)
        if not config.neuron.dont_wandb_log:
            wandb.log({"received_indices": synapse.dataset_indices})

        # Encode the dataset
        encoded_dataset = dataset.map(encode, batched=True)

        # Create a PyTorch DataLoader
        dataloader = DataLoader(encoded_dataset, batch_size=synapse.batch_size, collate_fn = default_data_collator)
        
        # Train data for one epoch
        for step, batch in enumerate(dataloader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            opt.zero_grad()

            # Forward pass
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                labels = labels
            )     
            # Backward pass    
            loss = outputs.loss
            if not config.neuron.dont_wandb_log:
                wandb.log({"loss":loss,
                            'opt_local_epoch':opt.local_epoch})
            loss.backward()
            # Adjust gradient
            opt.step()

            bt.logging.info(f"Step {step} Loss: {loss}")
            
        synapse.loss = loss

        bt.logging.info(f"Final Loss: {synapse.loss}")
        
        return synapse

    # Step 6: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=forward,
        blacklist_fn=blacklist,
        priority_fn=priority,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {priority, forward} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]} | "
                    f"Rank:{metagraph.R[my_subnet_uid]} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "            
                    f"Consensus:{metagraph.C[my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[my_subnet_uid]} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            # Check for auto update
            #if step % 30 == 0 and config.auto_update == "yes":
            #    util.try_update()
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config(neuron_type="miner"))
