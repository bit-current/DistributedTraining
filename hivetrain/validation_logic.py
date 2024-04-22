import torch
import time
import io
import math 
import threading
import os
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
import torch.nn as nn
import torch.nn.functional as F
import hashlib

class ModelValidator:
    def __init__(self, model, optimizer, data_loader,check_update_interval=300, bittensor_network = None, chain_manager= None,hf_manager=None, interval=300):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.interval = interval  # Validation interval in seconds
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.bittensor_network = bittensor_network
        self.scores = [0 for _ in range(len(self.bittensor_network.metagraph.hotkeys))]
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager
        self.last_pull_time = 0
        self.check_update_interval = check_update_interval

    
    # def receive_gradients(self, repo_id="your_username/your_repo_name", gradient_file_name="gradients.pt"):
    #     try:
    #         # Download the gradients file from Hugging Face Hub
    #         gradient_file_path = hf_hub_download(repo_id=repo_id, filename=gradient_file_name, use_auth_token=True)

    #         # Load the gradients directly using torch.load
    #         aggregated_gradients = torch.load(gradient_file_path)
                
    #         return aggregated_gradients
    #     except Exception as e:
    #         logging.debug(f"Error receiving gradients from Hugging Face: {e}")
    #         return None

    def update_model_weights(self, gradients, alpha=5e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= (gradients[name] * alpha)

    def evaluate_model(self, metric='perplexity'):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(self.data_loader): #FIXME turn me into a generator?
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
            
        logging.info("!Receiving Gradients from chain")
        self.bittensor_network.sync(lite=True)#FIXME too prone to issues 

        if time.time() - self.last_pull_time >= self.check_update_interval:
            if self.hf_manager.check_for_new_submissions(self.hf_manager.model_repo_id):
                logging.info("Averaged model updated on Hugging Face. Pulling latest model...")                
                self.hf_manager.pull_latest_model()
                time.sleep(10) #just to give enough time for pull
                self.model = self.hf_manager.update_model(self.model)
                optimizer = optim.Adam(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer
                self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()} 
            self.last_pull_time = time.time()

        self.original_state_dict = deepcopy(self.model.state_dict())

        for uid,hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
            hf_repo = self.chain_manager.retrieve_hf_repo(hotkey_address)
            gradients = self.hf_manager.receive_gradients(hf_repo)
            if gradients is not None:
                logging.info(f"Receiving Gradients from: {hotkey_address}")
                logging.info(f"Updating Model Weights")
                self.update_model_weights(gradients)
                logging.info(f"The model hash: {self.calculate_model_hash()}")
                logging.info(f"Evaluating model")
                loss, perplexity = self.evaluate_model()
                loss_score = max(0, self.base_loss - loss)
                perplexity_score = max(0, self.base_perplexity - perplexity)
                self.model.load_state_dict(self.original_state_dict)
            else:
                loss_score = 99999999
                perplexity_score = 99999999
                
            self.scores[uid] = perplexity_score

            # Reset the model to its original state
            
            try:
                logging.info(f"Loss: {loss}, Perplexity: {perplexity}")
            except:
                pass
            logging.info(f"Loss Score: {loss_score}, Perplexity Score: {perplexity_score}")
            time.sleep(0.1)


            #if self.bittensor_network.should_set_weights():    
            self.bittensor_network.set_weights(self.scores)

    def start_periodic_validation(self):
        #def run():
        
        while True:
            self.validate_and_score()
            logging.info(f"One round done sleeping for: {self.interval}")
            time.sleep(self.interval)
        
        #threading.Thread(target=run, daemon=True).start()

    def calculate_model_hash(self):
        
        model_hash = hashlib.sha256()
        for name, param in self.model.named_parameters():
            model_hash.update(name.encode('utf-8'))
            model_hash.update(param.data.cpu().numpy().tobytes())
        return model_hash.hexdigest()

class LocalValidator(ModelValidator):
    def __init__(self, model, optimizer, data_loader, bittensor_network=None, chain_manager=None,hf_manager=None, interval=3600, local_gradient_dir="local_gradients"):
        super().__init__(model, optimizer, data_loader, bittensor_network, chain_manager, hf_manager, interval)
        self.local_gradient_dir = local_gradient_dir
        # Ensure the local directory exists
        os.makedirs(self.local_gradient_dir, exist_ok=True)

    def receive_gradients(self, repo_id=None, gradient_file_name="gradients.pt"):
        """
        Overrides the receive_gradients method to fetch gradients from a local directory.
        """
        try:
            if repo_id == None:
                return None
            gradient_file_path = os.path.join(repo_id, gradient_file_name)
            if not os.path.exists(gradient_file_path):
                logging.warning(f"Gradient file not found: {gradient_file_path}")
                return None

            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)
            return aggregated_gradients
        except Exception as e:
            logging.error(f"Error receiving gradients locally: {e}")
            return None

class DeltaValidator(ModelValidator):

    def update_model_weights(self, weight_deltas, alpha=5e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weight_deltas:
                    param.data = weight_deltas[name] + param.data
    
class LocalDeltaValidator(DeltaValidator, LocalValidator):
    pass

class MNISTValidator(LocalValidator):
    def __init__(self, model, optimizer, data_loader, bittensor_network=None, chain_manager=None, hf_manager=None, interval=300,local_gradient_dir="local_gradients"):
        super().__init__(model, optimizer, data_loader, bittensor_network, chain_manager, hf_manager, interval)
        self.base_loss, self.base_accuracy = self.evaluate_model()  # Redefine to use accuracy for MNIST

    def evaluate_model(self, *args, **kwargs):
        """Evaluate the model on the MNIST validation dataset."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.data_loader:

                images, labels = batch
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item() 
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

class MNISTDeltaValidator(MNISTValidator):

    def update_model_weights(self, weight_deltas, alpha=5e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weight_deltas:
                    param.data = weight_deltas[name] + param.data
