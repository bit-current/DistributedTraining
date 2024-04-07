import torch
import time
import io
import math 
import threading
from bittensor import logging
from copy import deepcopy
from hivetrain.btt_connector import (
    BittensorNetwork,
    # get_validator_uids_and_addresses,
    serve_axon,
    sync
)

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class ModelValidator:
    def __init__(self, model, optimizer, data_loader, dht, bittensor_network = None, interval=3600):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.interval = interval  # Validation interval in seconds
        self.original_state_dict = deepcopy(model.state_dict())
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.scores = []
        self.bittensor_network = bittensor_network
        self.dht = dht

    @staticmethod
    def deserialize_gradients(serialized_gradients):
        buffer = io.BytesIO(serialized_gradients)
        buffer.seek(0)
        return torch.load(buffer)

    def receive_gradients(self, storage_dht = None, hotkey = None):
        try:
            serialized_gradients = storage_dht.get(hotkey).value
            aggregated_gradients = self.deserialize_gradients(serialized_gradients)
        except:
            return None
        
        return aggregated_gradients

    # def receive_gradients_from_chain(self):
    #     # Get validators uids
    #     if time.time() - self.last_sync_time > self.sync_interval:
    #         sync(self.last_sync_time, self.sync_interval, BittensorNetwork.config)#scope issue FIXME?
    #         self.last_sync_time = time.time()

    #     validator_uids = self.bittensor_network.get_validator_uids()
    #     # Get average of validator weights weighted by their stake?
    #     self.miner_gradients = []
    #     for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
    #         if uid not in validator_uids:
    #             try:
    #                 gradient = receive_gradients(self.dht, hotkey)
    #                 self.miner_gradients.append(gradient)
    #             except:
    #                 self.miner_gradients.append(None)

    def update_model_weights(self, gradients, alpha=0.00001):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= gradients[name] * alpha

    def evaluate_model(self, metric='loss'):
        #WARNING: # 2. The evaluate_model method incorrectly uses the criterion on outputs directly. 
        # If the model's outputs are logits, and the criterion expects logits and labels, this might be correct, 
        # but typically, a transformation is applied to outputs before calculating loss (e.g., softmax for cross-entropy loss).
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in self.data_loader:
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch["labels"])
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)

        average_loss = total_loss / total_samples
        perplexity = math.exp(average_loss) if metric == 'perplexity' else None
        return average_loss, perplexity

    def validate_and_score(self):
        ## Check if the model is changed on HF
            ## Check if HF commit hash is updated?
            ## If true pull please
            
        logging.info("Receiving Gradients from chain")
        for uid,hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
            logging.info(f"Receiving Gradients from: {hotkey_address}")
            gradients = self.receive_gradients(self.dht, hotkey_address)
            logging.info(f"Updating Model Weights")
            if gradients is not None:
                self.update_model_weights(gradients)
            logging.info(f"Evaluating model")
            loss, perplexity = self.evaluate_model()
            loss_score = max(0, self.base_loss - loss)
            perplexity_score = max(0, self.base_perplexity - perplexity) if perplexity else 0
            logging.info(f"Scoring")
            self.scores[uid] = perplexity_score

            # Reset the model to its original state
            logging.info("Reverting to base model")
            self.model.load_state_dict(self.original_state_dict)
            
            logging.info(f"Loss Score: {loss_score}, Perplexity Score: {perplexity_score}")
            time.sleep(0.1)

            self.bittensor_network.sync(lite=False)#FIXME too prone to issues 

            if self.bittensor_network.should_set_weights():    
                self.bittensor_network.set_weights(scores)

    def start_periodic_validation(self):
        #def run():
        while True:
            self.validate_and_score()
            time.sleep(self.interval)
        
        #threading.Thread(target=run, daemon=True).start()
