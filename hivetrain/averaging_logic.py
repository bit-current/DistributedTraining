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

from torch import nn, optim
import torch.nn.functional as F

from tqdm import tqdm

class Averager:
    def __init__(self, model, local_dir, repo_id,hf_manager,chain_manager,bittensor_network, hf_token=os.environ.get("HF_TOKEN")):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager

    def receive_gradients(self, repo_id="your_username/your_repo_name", gradient_file_name="gradients.pt"):
        try:
            # Download the gradients file from Hugging Face Hub
            gradient_file_path = hf_hub_download(repo_id=repo_id, filename=gradient_file_name, use_auth_token=True)

            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)

            if self.have_nans(aggregated_gradients):
                return None
                
            return aggregated_gradients
        except Exception as e:
            logging.debug(f"Error receiving gradients from Hugging Face: {e}")
            return None
    
    def receive_and_score_gradients(self):
        # Get validators uids
        self.bittensor_network.sync(lite=False)#scope issue FIXME?
            
        validator_uids = self.bittensor_network.get_validator_uids(vpermit_tao_limit=1024)

        if isinstance(self.bittensor_network.metagraph.W, list):
            self.validator_combined_weights = []
            weights_list = []
            for validator_uid in validator_uids:
                weights = self.bittensor_network.metagraph.W[validator_uid]
                if sum(weights)==0:
                    continue
                else:
                    weights_list.append(weights)
            self.validator_combined_weights = torch.mean(torch.tensor(weights_list), axis=0)
        else:
            self.validator_combined_weights = torch.mean(self.bittensor_network.metagraph.W[validator_uids, :], axis=0)
        #n = len(self.bittensor_network.metagraph.hotkeys) #FIXME I am only for testing NOPROD
        #self.validator_combined_weights = torch.full((n,1), 1/n, dtype=torch.float32) #FIXME I am only for testing NOPROD
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

    @staticmethod
    def have_nans(aggregated_gradients):
        for tensor in aggregated_gradients.values():
            if torch.isnan(tensor).any():
                logging.debug("NaN values detected in the aggregated gradients.")
                return True
        return False

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

            if self.hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = self.hf_manager.update_model(self.model)

            self.receive_and_score_gradients()
            averaged_gradients = self.average_gradients()
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)
            logging.info("Averaging Done")

class DeltaAverager(Averager):
    def __init__(self, model, local_dir, repo_id,hf_manager, chain_manager,bittensor_network, hf_token=os.environ.get("HF_TOKEN")):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager

    def average_gradients(self, beta=1.0):

        self.non_none_miner_gradients = [gradients for gradients in self.miner_gradients if gradients is not None]
        assert len(self.non_none_miner_gradients) > 0
        averaged_gradients = {name: torch.zeros_like(grad) for name, grad in self.non_none_miner_gradients[0].items()}

        for score, gradients in zip(self.validator_combined_weights, self.miner_gradients):
            logging.info("Averaging Gradient")
            if gradients is not None:
                for (name, grad), (param_name, param) in zip(gradients.items(), self.model.named_parameters()):
                    averaged_gradients[name] += (param + grad) * score * beta
        for name, param in averaged_gradients.items():
            averaged_gradients[name] = param / len(self.non_none_miner_gradients)
        return averaged_gradients

    def apply_averaged_gradients(self, averaged_gradients, alpha=0.00001):
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in averaged_gradients:
                    param = averaged_gradients[name]

    def run_periodic_averaging(self, t):
        while True:
            logging.info("Averaging Beggining")
            start_time = time.time()

            if self.hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = self.hf_manager.update_model(self.model)

            self.receive_and_score_gradients()
            averaged_weights = self.average_gradients()
            self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)
            logging.info("Averaging Done")

class LocalAverager(DeltaAverager):
    def __init__(self, model, local_dir,hf_manager, chain_manager, bittensor_network=None, hf_token=os.environ.get("HF_TOKEN")):
        super().__init__(model, local_dir,hf_manager=hf_manager, chain_manager=chain_manager,repo_id=None, bittensor_network=bittensor_network, hf_token=hf_token)
        # No need for repo_id or hf_token in the local version

    def receive_gradients(self, repo_id=None, gradient_file_name="gradients.pt"):
        """
        Overrides the receive_gradients method to fetch gradients from a local directory.
        """
        if repo_id is None:
            return None
        try:
            gradient_file_path = os.path.join(repo_id, gradient_file_name)
            if not os.path.exists(gradient_file_path):
                logging.warning(f"Gradient file not found: {gradient_file_path}")
                return None

            # Load the gradients directly using torch.load
            aggregated_gradients = torch.load(gradient_file_path)
            
            if self.have_nans(aggregated_gradients):
                return None

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
        os.makedirs(self.local_dir, exist_ok=True)
        model_save_path = os.path.join(self.local_dir, "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")


# class ParameterizedAveraging(nn.Module):
#     def __init__(self, model_paths):
#         super(ParameterizedAveraging, self).__init__()
#         self.model_paths = model_paths
#         self.num_models = len(model_paths)
#         averaging_model = ParameterizedAveraging(base_models).to(device)

#         self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)

#     def forward(self, x):
#         averaged_params = self.get_averaged_params()
#         output = self.forward_with_averaged_params(x, averaged_params)
#         return output

#     def get_averaged_params(self):
#         for params in self.lazy_load_params():
#             weighted_average = torch.zeros_like(params[0])
#             for param, weight in zip(params, self.weights):
#                 weighted_average += param * weight
#             yield weighted_average

#     def lazy_load_params(self):
#         for model_path in self.model_paths:
#             model = torch.load(model_path, map_location='cpu')
#             yield model.parameters()

#     def forward_with_averaged_params(self, x, averaged_params):
#         model = torch.load(self.model_paths[0], map_location='cpu')
#         for param, averaged_param in zip(model.parameters(), averaged_params):
#             param.data = averaged_param.data
#         model.to(x.device)
#         return model(x)

# class LocalParameterizedAverager(LocalAverager):
#     def meta_learning(model_paths, train_loader, val_loader, meta_epochs, inner_epochs, device):
#         cpu_device = torch.device("cpu")
#         for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
#             try:
#                 repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
#                 gradient = self.receive_gradients(repo_id=repo_id)
                
#             except Exception as e:
#                 logging.debug(f"Receiving gradients failed due to: {e}")

#         averaging_model = ParameterizedAveraging(model_paths).to(device)
#         meta_optimizer = optim.SGD(averaging_model.parameters(), lr=0.1)
#         criterion = nn.CrossEntropyLoss()

#         for epoch in range(meta_epochs):
            
#             # Outer loop: Update averaging weights
#             meta_optimizer.zero_grad()
            
#             # Perform averaging step on CPU
#             averaging_model.to(cpu_device)
#             averaged_params = list(averaging_model.get_averaged_params())
#             averaging_model.to(device)

#             val_loss = evaluate_model(averaging_model, val_loader, criterion, device)
#             val_loss.backward()
#             meta_optimizer.step()

#             print(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {val_loss:.4f}")

#         # Move averaging model back to CPU before returning
#         averaging_model.to(cpu_device)
#         return averaging_model

#     def run_periodic_averaging(self, t):
#         while True:
#             logging.info("Averaging Beggining")
#             start_time = time.time()
#             #self.receive_and_score_gradients()
#             averaged_weights = self.average_gradients()
#             self.apply_averaged_gradients(averaged_weights)
#             self.save_model()
#             self.push_to_hf_hub(commit_message="Updated model with new gradients")
#             elapsed_time = time.time() - start_time
#             time_to_wait = max(0, t - elapsed_time)
#             time.sleep(time_to_wait)
#             logging.info("Averaging Done")

class LocalParameterizedAverager(nn.Module, LocalAverager):
    def __init__(self, model,local_dir, device,hf_manager, chain_manager=None,bittensor_network=None, hf_token=os.environ.get("HF_TOKEN") ):
        nn.Module.__init__(self)
        LocalAverager.__init__(self,model, local_dir,hf_manager=hf_manager, chain_manager=chain_manager, bittensor_network=bittensor_network, hf_token=hf_token)
        #self.get_model_paths()
        #self.num_models = len(self.model_paths)
        #self.weights = nn.Parameter(torch.ones(self.num_models, device=device) / self.num_models)
        self.device = device

    def get_model_paths(self, gradient_file_name="gradients.pt"):
        self.model_paths = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                #gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None:
                    self.model_paths.append(repo_id)
            except Exception as e:
                logging.debug(f"Receiving gradients failed due to: {e}")

    def get_averaged_params(self):
        averaged_gradients = {name: torch.zeros_like(grad) for name, grad in self.model.named_parameters()}
        for params, weight in zip(self.lazy_load_params(), self.weights):
            for (name_weight_delta, param_weight_delta),(name_base_model, param_base_model) in zip(params.items(),self.model.named_parameters()):
                averaged_gradients[name_base_model] += (param_weight_delta + param_base_model) * weight #FIXME make weights per param

        return averaged_gradients

    def lazy_load_params(self):
        for model_path in self.model_paths:
            if model_path is None:
                yield None
            else:
                weight_delta = torch.load(os.path.join(model_path,"gradients.pt"), map_location='cpu')
                yield weight_delta

    def get_averaged_model(self):
        #model = torch.load(self.model_paths[0], map_location='cpu')
        averaged_params = self.get_averaged_params()
        for param, averaged_param in zip(self.model.parameters(), averaged_params.values()):
            param.data.copy_(averaged_param.data/self.num_models)  
        self.model.to(self.device)
        return self.model

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.local_dir, exist_ok=True)
        model_save_path = os.path.join(self.local_dir, "averaged_model.pt")
        torch.save(self.final_averaged_model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

    def meta_learning(self, val_loader, meta_epochs, lr):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([self.weights], lr=lr)

        for epoch in range(meta_epochs):
            # Outer loop: Update averaging weights
            
            
            #val_loss = evaluate_model(averaged_model, val_loader, criterion, self.device)

            #self.model.eval()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            
            for epoch in range(meta_epochs):
                for batch in val_loader:
                    averaged_model = self.get_averaged_model()

                    images, labels = batch
                    outputs = averaged_model(images)
                    val_loss = F.cross_entropy(outputs, labels)
                    total_loss += val_loss.item() 
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                    optimizer.zero_grad()
                    val_loss.backward()
                    optimizer.step()
                    logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {val_loss:.4f}, Weights: {self.weights}")

                average_loss = total_loss / total_samples
                accuracy = correct_predictions / total_samples
            
                
            

            # with torch.no_grad():
            #     self.weights /= self.weights.sum()

            

        return self.get_averaged_model()

    def run_periodic_averaging(self, val_loader,meta_epochs, t):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if self.hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = self.hf_manager.update_model(self.model)

            self.get_model_paths()
            self.num_models = len(self.model_paths)
            self.weights = nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))

            self.final_averaged_model = self.meta_learning(val_loader, meta_epochs, 0.001)
            #self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

class LocalParameterizedAverager(nn.Module):
    def __init__(self, model, local_dir, device, hf_manager, chain_manager=None, bittensor_network=None, hf_token=os.environ.get("HF_TOKEN")):
        super(LocalParameterizedAverager, self).__init__()
        self.model = model
        self.local_dir = local_dir
        self.device = device
        self.hf_manager = hf_manager
        self.chain_manager = chain_manager
        self.bittensor_network = bittensor_network
        self.hf_token = hf_token
        self.population_size = 10
        self.num_generations = 10
        self.sigma = 0.1  # Standard deviation for Gaussian noise

    def get_model_paths(self):
        self.model_paths = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                #gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None:
                    self.model_paths.append(repo_id)
            except Exception as e:
                logging.debug(f"Receiving gradients failed due to: {e}")

    def lazy_load_params(self):
        for model_path in self.model_paths:
            if model_path is None:
                yield None
            else:
                weight_delta = torch.load(os.path.join(model_path,"gradients.pt"), map_location='cpu')
                yield weight_delta

    def get_averaged_params(self, weights):
        averaged_params = {name: torch.zeros_like(param, device=self.device) for name, param in self.model.named_parameters()}
        for params, weight in zip(self.lazy_load_params(), weights):
            for (name_weight_delta, param_weight_delta), (name_base_model, param_base_model) in zip(params.items(), self.model.named_parameters()):
                averaged_params[name_base_model] += (param_weight_delta + param_base_model) * weight
        return averaged_params

    def get_averaged_model(self, weights):
        averaged_params = self.get_averaged_params(weights)
        for name, param in self.model.named_parameters():
            param.data.copy_(averaged_params[name].data / len(self.model_paths))
        return self.model

    def evaluate_population(self, val_loader, population):
        # Evaluate all individuals in the population
        fitness_scores = []
        for weights in tqdm(population):
            model = self.get_averaged_model(weights)
            total_loss = 0
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
            average_loss = total_loss / len(val_loader)
            fitness_scores.append(-average_loss)  # Negative because lower loss is better
        return fitness_scores

    def evolve_population(self, population,val_loader):
        fitness_scores = self.evaluate_population(val_loader, population)
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order of fitness
        best_individuals = [population[i] for i in sorted_indices[:len(population)//2]]  # Select top 50%
        
        # Reproduce with mutation
        new_population = []
        while len(new_population) < len(population):
            parent = random.choice(best_individuals)
            child = parent + torch.randn_like(parent) * self.sigma  # Gaussian mutation
            new_population.append(child)
        return new_population

    def run_evolution(self, val_loader):
        # Initialize population
        population = [torch.rand(len(self.model_paths), device=self.device) for _ in range(self.population_size)]
        
        for generation in tqdm(range(self.num_generations)):
            population = self.evolve_population(population,val_loader)
            best_weights = population[0]
            best_fitness = self.evaluate_population(val_loader, [best_weights])[0]
            print(f"Generation {generation}, Best Fitness: {best_fitness}")

        return self.get_averaged_model(best_weights)

    def run_periodic_averaging(self, val_loader, t=40):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if self.hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = self.hf_manager.update_model(self.model)

            self.get_model_paths()
            self.num_models = len(self.model_paths)
            self.weights = nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))

            self.final_averaged_model = self.run_evolution(val_loader)
            #self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")
        

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