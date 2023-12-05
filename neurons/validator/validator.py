# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

# Bittensor Validator Template:
# TODO(developer): Rewrite based on protocol defintion.

import argparse
import asyncio

# test pretrain
import math

# Step 1: Import necessary libraries and modules
import os
import random
import time
import traceback

import bittensor as bt
import hivemind
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from optimum.bettertransformer import BetterTransformer
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import AsyncDendritePool, get_random_uids
from validator_core import DatasetStateSingelton, ModelSingleton, upload_checkpoint

# import this repo
import template


# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():

    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/miner.py --help
    config =  bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'validator',
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config

async def main( config ):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.info(f"Metagraph: {metagraph}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite( wallet = wallet )
    dendrite_pool = AsyncDendritePool( wallet = wallet, metagraph = metagraph )
    bt.logging.info(f"Dendrite: {dendrite}")

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
    alpha = 0.9
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    
    # Step 7: Init DHT
    initial_peers= "/ip4/161.97.156.125/tcp/8108/p2p/12D3KooWRYFcmYhf7Lyn3TH9cFxqhruhnjMYky4ehdZHFuv6M4A9"
    model_name: str = "kmfoda/tiny-random-gpt2"
    lr = 0.00001
    dht = hivemind.DHT(initial_peers=[initial_peers], start=True)
    print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

    # Step 7: Init dataset
    config.dataset_name = "wikitext"
    config.batch_size = 16
    config.num_of_duplicates = 2 # number of miners running the same process for validation
    config.run_id = 's25_test_run'
    config.upload_interval = 900
    config.weight_update_interval = 900

    latest_upload = 0
    latest_weight_update = 0

    dataset = load_dataset(config.dataset_name, 'wikitext-2-v1', split='train')
    dataset_indices = [i for i in range(0, len(dataset))]
    dataset_common_state = DatasetStateSingelton(dht ,dataset_indices)

    ## Do I need to wait before I push the model to the GPU after hivemindopt?
    model = ModelSingleton.get_instance(model_name)
    
    opt = torch.optim.AdamW(model.parameters(), lr = lr)

    breakpoint()
    opt = hivemind.Optimizer(
        dht=dht,                  # use a DHT that is connected with other peers
        run_id=config.run_id,    # unique identifier of this collaborative run
        batch_size_per_step=32,   # each call to opt.step adds this many samples towards the next epoch
        target_batch_size=10000,  # after peers collectively process this many samples, average weights and begin the next epoch 
        optimizer=opt,            # wrap the SGD optimizer defined above
        use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
        matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
        averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
        verbose=True              # print logs incessently
    )

    state_averager = TrainingStateAverager(
                    dht=dht,
                    optimizer=opt,
                    scheduler=get_linear_schedule_with_warmup(opt, num_warmup_steps=5000, num_training_steps=125_000),
                    prefix=f"{config.run_id}_state_averager",
                    state_compression=hivemind.Float16Compression(),
                    bandwidth=optimizer_args.bandwidth,
                    client_mode=optimizer_args.client_mode,
                    start=True,
                    **asdict(averager_args),
    )

    # Step 8: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    while True:
        try:
            
            uids = get_random_uids(metagraph, k=100) # .to(self.device)
            total_per_pass = 100000
            examples_per_uid = total_per_pass // len(uids)
            uids_indices = dataset_common_state.get_dataset_indices(len(uids), examples_per_uid)
            
            # TODO split queries
            queries = []
            for uid_idx in range(0, len(uids), examples_per_uid):
                
                shuffled_uid_group = random.shuffle(uids_indices[uid_idx:uid_idx+examples_per_uid])
                #TODO get the hashes of the groups
                # Make miners return the group hashes with their losses as well.
                queries.append(
                    template.train.Train( 
                        dataset_indices=shuffled_uid_group,
                        model_name = "kmfoda/tiny-random-gpt2", 
                        dht_connector = dht_state.get_visible_maddrs(), # TODO Add a decorator or sth for this to get the values 
                        batch_size = config.batch_size #TODO let miners decide this? Based on their hardware?
                    )
                )

            
            breakpoint()

            responses = await dendrite_pool.async_forward(
                uids,
                queries
            )

            breakpoint()
            # Log the results for monitoring purposes.
            bt.logging.info(f"Received responses: {responses}")

            if step % 1 == 0:
                ### TODO add anomaly detection here
                # swarm_responses = responses[(i*config.num_of_duplicates):(i*config.num_of_duplicates)+(config.num_of_duplicates+1)]
                # swarm_losses = [swarm_response.loss for swarm_response in swarm_responses]
                ###

                if time.time() - latest_upload > config.upload_interval:
                    state_averager.load_state_from_peers()
                    upload_checkpoint(commit_message, state_averager, model, repo_path,repo_url)
                    latest_upload = time.time()

                if time.time() - latest_weight_update > config.weight_update_interval:
                    state_averager.load_state_from_peers()
                    upload_checkpoint(commit_message, state_averager, model, repo_path,repo_url)
                    latest_weight_update = time.time()

                # Initialize the score for the current miner's response.
                score = np.zeros(len(responses))

                # # Use CUDA if available, otherwise use CPU
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load pre-trained model and tokenizer
                # model_name = 'sshleifer/tiny-gpt2'
                # response = ([], "kmfoda/tiny-random-gpt2", 'wikitext', 4, None)

                # load model weights
                for layer, weight in zip(model.parameters(), response.model_weights):
                    # layer = torch.nn.parameter.Parameter(weight)
                    layer = torch.nn.parameter.Parameter(bt.Tensor.deserialize(weight).clone().detach())

                tokenizer = AutoTokenizer.from_pretrained(response.model_name)
                
                # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
                tokenizer.pad_token = tokenizer.eos_token

                # Move the model to the appropriate device
                model.to(device)

                # Load optimized and scheduler
                if response.optimizer_name == "adam":
                    optimizer = torch.optim.AdamW(model.parameters(), lr = response.lr)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr = response.lr)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=response.steps)  

                # Define encoding function
                def encode(examples):
                    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

                # Select the correct datapoints
                dataset_sample = dataset.select(response.dataset_indices)

                # Encode the dataset
                encoded_dataset = dataset_sample.map(encode, batched=True)

                # Create a PyTorch DataLoader
                dataloader = DataLoader(encoded_dataset, batch_size=response.batch_size)

                # if response.gradients == []:
                #     scores[i] = 0
                #     continue
                # else:
                #     for layer, new_grads in zip(model.named_parameters(),response.gradients):
                #         # layer[1].grad = torch.tensor(bt.Tensor.deserialize(new_grads))
                #         layer[1].grad = bt.Tensor.deserialize(new_grads).clone().detach()
                    
                #     # Adjust gradient
                #     optimizer.step()
                #     scheduler.step() 
                #     optimizer.zero_grad()

                # Train data for one epoch
                for step, batch in enumerate(dataloader):
                    
                    # Move batch to device
                    input_ids = torch.stack(batch['input_ids']).to(device)
                    attention_mask = torch.stack(batch['attention_mask']).to(device)
                    labels = torch.stack(batch["input_ids"]).to(device)

                    # Forward pass
                    outputs = model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask,
                        labels = labels
                    )     
                    
                    # Backward pass
                    loss = outputs.loss
                    print(step)
                    print(loss)
                    # synpase.loss = loss
                    loss.backward()

                    # Adjust gradient
                    optimizer.step()
                    scheduler.step() 
                    optimizer.zero_grad()

                    if step == 10:
                        break

                    outputs = model(
                        input_ids = torch.stack(batch["input_ids"]).to(device), 
                        attention_mask = torch.stack(batch["attention_mask"]).to(device),
                        labels = torch.stack(batch["input_ids"]).to(device)
                    )  

                    print("final loss")
                    correct_loss = float(outputs.loss)

                    for swarm_response in swarm_responses:
                        rmse = math.sqrt(np.square(np.subtract([correct_loss],[response.loss])).mean())
                        # Check if the miner has provided the correct response by doubling the dummy input.
                        # If correct, set their score for this round to 1.
                        if rmse < 0.01:
                            score = 1

                        # Update the global score of the miner.
                        # This score contributes to the miner's weight in the network.
                        # A higher weight means that the miner has been consistently responding correctly.
                        scores[i] = alpha * scores[i] + (1 - alpha) * score

                        # Log the results for monitoring purposes.
                        bt.logging.info(f"Score: {score}")

            # Periodically update the weights on the Bittensor blockchain.
            # NOTE Disbaled for now due to weight settting bug
            if (step + 1) % 10000000000 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                # breakpoint()
                result = subtensor.set_weights(
                    netuid = config.netuid, # Subnet to set weights on.
                    wallet = wallet, # Wallet to sign set weights using hotkey.
                    uids = metagraph.uids, # Uids of the miners to set weights for.
                    weights = weights, # Weights to set for the miners.
                    wait_for_inclusion = True
                )
                if result: bt.logging.success('Successfully set weights.')
                else: bt.logging.error('Failed to set weights.') 

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    asyncio.run( main(config) )
