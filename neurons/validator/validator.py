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

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt
import random
import asyncio

# import this repo
import template

# test pretrain
import math
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from optimum.bettertransformer import BetterTransformer
from utils import get_random_uids, AsyncDendritePool

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
    
    # Step 7: Init dataset
    config.dataset_name = "wikitext"
    config.batch_size = 16
    config.num_of_duplicates = 2 # number of miners running the same process for validation
    dataset = load_dataset(config.dataset_name, 'wikitext-2-v1', split='train')

    # Step 8: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    while True:
        try:
            
            uids = get_random_uids(metagraph, k=100) # .to(self.device)
            num_data_splits = math.floor(len(uids) / config.num_of_duplicates)
            split_uids = [uids[(i*num_data_splits):(i*num_data_splits) + (num_data_splits+1)] for i in range(0, num_data_splits)]  
            
            # axons = [metagraph.axons[uid] for uid in uids]
            # split_axons = [[metagraph.axons[uid] for uid in uids] for uids in split_uids]
            
            # TODO strucutre in a way so that we don't repeat data points
            start_index = random.randint(0, len(dataset)-(config.batch_size*num_data_splits))
            end_index = start_index + (config.batch_size*num_data_splits)

            # TODO split queries
            queries = []
            for i in range(0, len(split_uids)):
                queries.append(
                    template.train.Train( 
                        dataset_indices=[start_index * (config.batch_size*i), (end_index*(config.batch_size*i)) + (config.batch_size*i)],  
                        model_name = "kmfoda/tiny-random-gpt2", 
                        batch_size = config.batch_size
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

            # TODO(developer): Define how the validator scores responses.
            # Adjust the scores based on responses from miners.
            for i, resp_i in enumerate(responses):
                # Initialize the score for the current miner's response.
                score = 0
            
                # # Use CUDA if available, otherwise use CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load pre-trained model and tokenizer
                # model_name = 'sshleifer/tiny-gpt2'
                # resp_i = ([], "kmfoda/tiny-random-gpt2", 'wikitext', 4, None)
                model = AutoModelForCausalLM.from_pretrained(resp_i.model_name)

                # load model weights
                for layer, weight in zip(model.parameters(), resp_i.model_weights):
                    # layer = torch.nn.parameter.Parameter(weight)
                    layer = torch.nn.parameter.Parameter(bt.Tensor.deserialize(weight).clone().detach())

                tokenizer = AutoTokenizer.from_pretrained(resp_i.model_name)
                
                # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
                tokenizer.pad_token = tokenizer.eos_token

                # Move the model to the appropriate device
                model.to(device)

                # Load optimized and scheduler
                if resp_i.optimizer_name == "adam":
                    optimizer = torch.optim.AdamW(model.parameters(), lr = resp_i.lr)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr = resp_i.lr)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=resp_i.steps)  

                # Define encoding function
                def encode(examples):
                    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

                # Select the correct datapoints
                dataset_sample = dataset.select(range(start_index, end_index))

                # Encode the dataset
                encoded_dataset = dataset_sample.map(encode, batched=True)

                # Create a PyTorch DataLoader
                dataloader = DataLoader(encoded_dataset, batch_size=resp_i.batch_size)

                if resp_i.gradients == []:
                    scores[i] = 0
                    continue
                else:
                    for layer, new_grads in zip(model.named_parameters(),resp_i.gradients):
                        # layer[1].grad = torch.tensor(bt.Tensor.deserialize(new_grads))
                        layer[1].grad = bt.Tensor.deserialize(new_grads).clone().detach()
                    
                    # Adjust gradient
                    optimizer.step()
                    scheduler.step() 
                    optimizer.zero_grad()

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
                    loss = float(outputs.loss)
                    rmse = math.sqrt(np.square(np.subtract([loss],[resp_i.loss])).mean())

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
