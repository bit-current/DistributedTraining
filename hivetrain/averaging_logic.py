import time
import torch
import random
import math
import copy
import torch
from huggingface_hub import Repository, HfFolder
from copy import deepcopy 
from hivemind.btt_connector import BittensorNetwork, sync
import io

class Averager:
    def __init__(self, model, local_dir, repo_id,dht,bittensor_network, hf_token=None):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.dht = dht

    @staticmethod
    def deserialize_gradients(serialized_gradients):
        buffer = io.BytesIO(serialized_gradients)
        buffer.seek(0)
        return torch.load(buffer)

    @staticmethod
    def receive_gradients(storage_dht = dht_manager.my_dht, hotkey = my_hotkey):
        serialized_gradients = storage_dht.get(hotkey)
        aggregated_gradients = deserialize_gradients(serialized_gradients)
        
        return aggregated_gradients
    
    def receive_and_score_gradients(self):
        # Get validators uids
        if time.time() - self.last_sync_time > self.sync_interval:
            sync(self.last_sync_time, self.sync_interval, BittensorNetwork.config)#scope issue FIXME?
            self.last_sync_time = time.time()

        validator_uids = self.bittensor_network.get_validator_uids()
        validator_combined_weights = torch.mean(self.bittensor_network.metagraph.W[validator_uids, :],axis=0)
        # Get average of validator weights weighted by their stake?
        self.miner_gradients = []
        self.miner_weights = []
        self.miner_hotkeys = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            if uid not in validator_uids:
                try:
                    gradient = receive_gradients(self.dht, hotkey)
                    miner_gradients.append(miner_gradients)
                except:
                    self.miner_gradients.append(None)
                    self.miner_weights.append(0.0)
                    self.miner_hotkeys.append(hotkey)
        

    def average_gradients(self, beta=1.0):
        averaged_gradients = {name:torch.zeros_like(grad) for name, grad in self.miner_gradients.items()}

        for score, gradients in zip(self.miner_scores, self.miner_gradients):
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
        if self.hf_token is not None:
            HfFolder.save_token(self.hf_token)
        
        repo = Repository(local_dir=self.local_dir, repo_id=self.repo_id, clone_from=f"https://huggingface.co/{self.repo_id}")
        repo.push_to_hub(commit_message=commit_message)

    def run_periodic_averaging(self, t, scored_gradients):
        while True:
            start_time = time.time()
            receive_and_score_gradients
            averaged_gradients = self.average_gradients(scored_gradients)
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)


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
            start_time = time.time()
            receive_and_score_gradients
            averaged_gradients = self.average_gradients(scored_gradients)
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)