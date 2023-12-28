# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
import torch
import asyncio
import threading
import traceback

import bittensor as bt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from torch.utils.data import DataLoader
import hivemind
from functools import partial

from template.base.neuron import BaseNeuron


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        
        self.forward_event = asyncio.Event()    
        
        # Init device
        self.device = self.config.neuron.device

        # Init DHT and model
        dht = hivemind.DHT(initial_peers=[self.config.neuron.initial_peers], start=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)
        
        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Set up a decentralized optimizer that will average with peers in background
        opt = torch.optim.AdamW(self.model.parameters(), lr = self.config.neuron.lr)
        self.opt = hivemind.Optimizer(
            dht=dht,                    # use a DHT that is connected with other peers
            run_id=self.config.neuron.run_id,        # unique identifier of this collaborative run
            scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
            batch_size_per_step=self.config.neuron.batch_size_train,     # each call to opt.step adds this many samples towards the next epoch
            target_batch_size=self.config.neuron.target_batch_size,    # after peers collectively process this many samples, average weights and begin the next epoch
            optimizer=opt,              # wrap the SGD optimizer defined above
            use_local_updates=True,     # perform optimizer steps with local gradients, average parameters in background
            matchmaking_time=10.0,       # when averaging parameters, gather peers in background for up to this many seconds
            averaging_timeout=10.0,     # give up on averaging if not successful in this many seconds
            verbose=False               # print logs incessently
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        self.dataset = load_dataset(self.config.neuron.dataset_name, 'wikitext-2-v1', split='train')
        

    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')    
                

    async def training_routine(self):
        print("Starting training routine")  # Debug print

        # Encode the dataset
        encoded_dataset = self.dataset.map(self.encode, batched=True)
        # Create a PyTorch DataLoader
        dataloader = DataLoader(encoded_dataset, batch_size=self.config.neuron.batch_size_train, collate_fn = default_data_collator)
        
        while not self.should_exit:
            
            # Train data for one epoch
            for step, batch in enumerate(dataloader):                  

                if self.forward_event.is_set():  # Check if the event is set
                    print("Awaiting forward event")
                    self.opt.tracker.pause_updates()
                    await self.forward_event.wait()  # Only wait if the event is set    
                    #await asyncio.sleep(1)
                    #self.opt.load_state_from_peers()

                try:
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = input_ids.clone()

                    self.opt.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask,
                        labels = labels
                        )    
                    
                    # Backward pass    
                    loss = outputs.loss
                    
                    if not self.config.neuron.dont_wandb_log:
                        self.wandb.log({"loss":loss, 'opt_local_epoch':self.opt.local_epoch})
                        
                    loss.backward()
                    self.opt.step()

                    bt.logging.info(f"Step {step} Loss: {loss}")
                except Exception as e:
                    bt.logging.error(f"Training error at step {step}: {e}")
            
            

    async def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()
        bt.logging.info(f"Miner starting at block: {self.block}")
        
        print("Creating training task")  # Debug print
        # Start training_routine as a background task
        training_task = asyncio.create_task(self.training_routine())

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    print("running while")
                    
                    # Wait before checking again.
                    #time.sleep(1)
                    await asyncio.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1
                
            # Await the training task to ensure it completes before exiting
            await training_task
        
        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.should_exit = True
            await training_task
            self.opt.shutdown()
            self.dht.shutdown()
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            await training_task

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
           
            self.is_running = False
            bt.logging.debug("Stopped")

    async def __aenter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        # self.run_in_background_thread()
        await self.run()
        return self
    

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def set_weights(self):
        """
        Self-assigns a weight of 1 to the current miner (identified by its UID) and
        a weight of 0 to all other peers in the network. The weights determine the trust level the miner assigns to other nodes on the network.

        Raises:
            Exception: If there's an error while setting weights, the exception is logged for diagnosis.
        """
        try:
            # --- query the chain for the most current number of peers on the network
            chain_weights = torch.zeros(
                self.subtensor.subnetwork_n(netuid=self.metagraph.netuid)
            )
            chain_weights[self.uid] = 1

            # --- Set weights.
            self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.metagraph.netuid,
                uids=torch.arange(0, len(chain_weights)),
                weights=chain_weights.to("cpu"),
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )

        except Exception as e:
            bt.logging.error(
                f"Failed to set weights on chain with exception: { e }"
            )

        bt.logging.info(f"Set weights: {chain_weights}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
    
    def save_state(self):
        """Saves the state of the validator to a file."""
        ...

    def load_state(self):
        """Loads the state of the validator from a file."""
        ...
