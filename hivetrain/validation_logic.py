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
    def __init__(self, model, optimizer, data_loader, bittensor_network = None, chain_manager= None, interval=3600):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.interval = interval  # Validation interval in seconds
        self.original_state_dict = deepcopy(model.state_dict())
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.bittensor_network = bittensor_network
        self.scores = [0 for _ in range(len(self.bittensor_network.metagraph.hotkeys))]
        self.chain_manager = chain_manager

    def receive_gradients(self, repo_id="your_username/your_repo_name", gradient_file_name="gradients.pt"):
        try:
            # Download the gradients file from Hugging Face Hub
            gradient_file_path = hf_hub_download(repo_id=repo_id, filename=gradient_file_name, use_auth_token=True)

            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)
                
            return aggregated_gradients
        except Exception as e:
            logging.debug(f"Error receiving gradients from Hugging Face: {e}")
            return None

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
        self.bittensor_network.sync(lite=True)#FIXME too prone to issues 

        for uid,hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):

            logging.info(f"Receiving Gradients from: {hotkey_address}")
            hf_repo = self.chain_manager.retreive_hf_repo(hotkey_address)
            gradients = self.receive_gradients(hf_repo)
            logging.info(f"Updating Model Weights")
            if gradients is not None:
                self.update_model_weights(gradients)
            logging.info(f"Evaluating model")
            loss, perplexity = self.evaluate_model()
            loss_score = max(0, self.base_loss - loss)
            perplexity_score = max(0, self.base_perplexity - perplexity) if perplexity else 0
            self.scores[uid] = perplexity_score

            # Reset the model to its original state
            
            self.model.load_state_dict(self.original_state_dict)
            if gradients is not None:
                logging.info(f"Loss Score: {loss_score}, Perplexity Score: {perplexity_score}")
            time.sleep(0.1)


            if self.bittensor_network.should_set_weights():    
                self.bittensor_network.set_weights(self.scores)

    def start_periodic_validation(self):
        #def run():
        while True:
            self.validate_and_score()
            time.sleep(self.interval)
        
        #threading.Thread(target=run, daemon=True).start()

class LocalValidator(ModelValidator):
    def __init__(self, model, optimizer, data_loader, bittensor_network=None, chain_manager=None, interval=3600, local_gradient_dir="local_gradients"):
        super().__init__(model, optimizer, data_loader, bittensor_network, chain_manager, interval)
        self.local_gradient_dir = local_gradient_dir
        # Ensure the local directory exists
        os.makedirs(self.local_gradient_dir, exist_ok=True)

    def receive_gradients(self, repo_id=None, gradient_file_name="gradients.pt"):
        """
        Overrides the receive_gradients method to fetch gradients from a local directory.
        """
        try:
            gradient_file_path = os.path.join(self.local_gradient_dir, gradient_file_name)
            if not os.path.exists(gradient_file_path):
                logging.warning(f"Gradient file not found: {gradient_file_path}")
                return None

            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)
            return aggregated_gradients
        except Exception as e:
            logging.error(f"Error receiving gradients locally: {e}")
            return None