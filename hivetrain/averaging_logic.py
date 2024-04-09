import time
import torch
import random
import math
import copy
import torch
from huggingface_hub import Repository, HfFolder
from copy import deepcopy 
from hivetrain.btt_connector import BittensorNetwork, sync
#from hivetrain.dht_connector import DHTManager
import io
from bittensor import logging
import os 
from transformers import TrainingArguments, Trainer


class Averager:
    def __init__(self, model, local_dir, repo_id,chain_manager,bittensor_network, hf_token=os.environ.get("HF_TOKEN")):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
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
    
    def receive_and_score_gradients(self):
        # Get validators uids
        self.bittensor_network.sync(lite=False)#scope issue FIXME?
            
        validator_uids = self.bittensor_network.get_validator_uids(vpermit_tao_limit=1024)
        #breakpoint()
        #validator_combined_weights = torch.mean(self.bittensor_network.metagraph.W[validator_uids, :],axis=0)
        n = len(self.bittensor_network.metagraph.hotkeys) #FIXME I am only for testing NOPROD
        self.validator_combined_weights = torch.full((n,1), 1/n, dtype=torch.float32) #FIXME I am only for testing NOPROD
        # Get average of validator weights weighted by their stake?
        self.miner_gradients = []
        self.miner_weights = []
        self.miner_hotkeys = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                gradient = self.receive_gradients(repo_id=repo_id)
                self.miner_gradients.append(gradient)
            except Exception as e:
                logging.debug(f"Receiving gradients failed due to: {e}")
                self.miner_gradients.append(None)
                self.miner_weights.append(0.0)
                self.miner_hotkeys.append(hotkey)
        

    def average_gradients(self, beta=1.0):

        self.miner_gradients = [gradients for gradients in self.miner_gradients if gradients is not None]
        assert len(self.miner_gradients) > 0
        averaged_gradients = {name: torch.zeros_like(grad) for name, grad in self.miner_gradients[0].items()}

        for score, gradients in zip(self.validator_combined_weights, self.miner_gradients):
            logging.info("Averaging Gradient")
            for name, grad in gradients.items():
                averaged_gradients[name] += grad * score * beta

        return averaged_gradients

    def apply_averaged_gradients(self, averaged_gradients, alpha=0.00001):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in averaged_gradients:
                    param -= alpha * averaged_gradients[name]

    def save_model(self):
        self.model.save_pretrained(self.local_dir)

    def push_to_hf_hub(self, commit_message="Pushing model to Hub"):
        training_args = TrainingArguments(
        output_dir=self.local_dir,  # Local directory to save the model
        per_device_train_batch_size=1,  # Dummy argument, won't actually be used for training here
        per_device_eval_batch_size=1,   # Dummy argument, necessary to specify but won't be used
        push_to_hub=True,               # Enable pushing to hub
        push_to_hub_model_id=self.repo_id,  # Repository ID on the Hugging Face Hub
        push_to_hub_organization=None,  # Specify organization name here if applicable
        push_to_hub_token=self.hf_token,  # Hugging Face authentication token
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,  # Your PyTorch model
            args=training_args,
        )

        # Push the model to the Hugging Face Hub
        trainer.push_to_hub(commit_message=commit_message)

    def run_periodic_averaging(self, t):
        while True:
            logging.info("Averaging Beggining")
            start_time = time.time()
            self.receive_and_score_gradients()
            averaged_gradients = self.average_gradients()
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)
            logging.info("Averaging Done")




class LocalAverager(Averager):
    def __init__(self, model, local_dir, dht=None, bittensor_network=None):
        super().__init__(model, local_dir, repo_id=None, dht=dht, bittensor_network=bittensor_network)
        # No need for repo_id or hf_token in the local version

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

    def push_to_hf_hub(self, commit_message="Pushing model to Hub"):
        """
        Overrides the push_to_hf_hub method to simply save the model locally.
        """
        self.save_model()
        logging.info(f"Model saved locally in {self.local_dir} instead of pushing to Hugging Face Hub.")
    
    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        model_save_path = os.path.join(self.local_dir, "saved_model")
        os.makedirs(model_save_path, exist_ok=True)
        self.model.save_pretrained(model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

# Example usage
if __name__ == "__main__":
    # Assuming model, local_dir, dht, and bittensor_network are defined
    local_averager = LocalAverager(model=model, local_dir="./local_models")
    # You can now use local_averager for operations like receive_gradients and apply_averaged_gradients


##FIXME where does it get its scored_gradients
class RankedAverager(Averager):
    def __init__(self, model, local_dir, repo_id, hf_token=None, validator=None, population_size=10, generations=5, mutation_rate=0.1, top_k=20):
        super().__init__(model, local_dir, repo_id, hf_token)
        self.population_size = population_size
        self.generations = generations
        self.top_k = top_k  # Number of top performing gradients to consider

    def generate_initial_population(self, gradients, scores):
        # Sort gradients based on scores, and take the top_k
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        top_k_gradients = [gradients[i] for i in top_k_indices]
        population = []
        for _ in range(self.population_size):
            individual = self.combine_gradients(top_k_gradients)
            population.append(individual)
        return population

    def combine_gradients(self, top_k_gradients):
        combined_gradient = {}
        for grad_dict in top_k_gradients:
            for name, grad in grad_dict.items():
                if name not in combined_gradient:
                    combined_gradient[name] = grad.clone()  # Use clone to ensure no inplace operation
                else:
                    if random.random() < 0.5:
                        combined_gradient[name] = combined_gradient[name] + grad
        return combined_gradient

    def evaluate_individual(self, individual):
        original_state_dict = copy.deepcopy(self.model.state_dict())
        self.apply_averaged_gradients(individual)
        loss, _ = self.model_validator.evaluate_model()
        self.model.load_state_dict(original_state_dict)  # Restore original state
        return loss

    def rank_selection(self, population):
        ranked = sorted(population, key=self.evaluate_individual)
        return ranked

    def average_gradients(self, gradients, scores):
        population = self.generate_initial_population(gradients, scores)

        for generation in range(self.generations):
            ranked_population = self.rank_selection(population)
            new_population = ranked_population[:self.top_k]  # Keep top_k performers

            # Generate new individuals by combining top performers
            while len(new_population) < self.population_size:
                selected = random.sample(new_population, 2)
                new_individual = self.combine_gradients(selected)
                new_population.append(new_individual)

            population = new_population

        # Returning the best gradients after final generation
        best_gradients = self.rank_selection(population)[0]
        return best_gradients

    def run_periodic_averaging(self, t, scored_gradients):
        while True:
            logging.info("Averaging Beggining")
            start_time = time.time()
            receive_and_score_gradients
            averaged_gradients = self.average_gradients(self.miner_gradients, self.miner_scores)
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            logging.info("Pushing to HF")
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)