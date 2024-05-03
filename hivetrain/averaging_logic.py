import copy
import os
import time
import torch
import random
import math
import mlflow
import mlflow.pytorch
from huggingface_hub import hf_hub_download
from hivetrain.config.mlflow_config import MLFLOW_UI_URL, CURRENT_MODEL_NAME
from hivetrain.utils.mlflow_utils import VERSION, initialize_mlflow, log_model_metrics
from hivetrain.config.mlflow_config import (
    MLFLOW_UI_URL,
    CURRENT_MODEL_NAME,
    MLFLOW_ACTIVE,
)

from copy import deepcopy
from hivetrain.btt_connector import BittensorNetwork, sync
from bittensor import logging
from transformers import TrainingArguments, Trainer, AdamW
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Averager:
    def __init__(
        self,
        model,
        local_dir,
        repo_id,
        hf_manager,
        chain_manager,
        bittensor_network,
        hf_token=os.environ.get("HF_TOKEN"),
    ):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager


        # initialize mlflow
        if MLFLOW_ACTIVE:
                initialize_mlflow(
                    role="averager",
                    device=self.device,
                    version=VERSION,
                    mlflow_ui_url=MLFLOW_UI_URL,
                    current_model_name=CURRENT_MODEL_NAME)



    def receive_gradients(
        self, repo_id="your_username/your_repo_name", gradient_file_name="gradients.pt"
    ):
        try:
            # Download the gradients file from Hugging Face Hub
            gradient_file_path = hf_hub_download(
                repo_id=repo_id, filename=gradient_file_name, use_auth_token=True
            )

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
        self.bittensor_network.sync(lite=False)  # scope issue FIXME?

        validator_uids = self.bittensor_network.get_validator_uids(
            vpermit_tao_limit=1024
        )

        if isinstance(self.bittensor_network.metagraph.W, list):
            self.validator_combined_weights = []
            weights_list = []
            for validator_uid in validator_uids:
                weights = self.bittensor_network.metagraph.W[validator_uid]
                if sum(weights) == 0:
                    continue
                else:
                    weights_list.append(weights)
            self.validator_combined_weights = torch.mean(
                torch.tensor(weights_list), axis=0
            )
        else:
            self.validator_combined_weights = torch.mean(
                self.bittensor_network.metagraph.W[validator_uids, :], axis=0
            )
        # n = len(self.bittensor_network.metagraph.hotkeys) #FIXME I am only for testing NOPROD
        # self.validator_combined_weights = torch.full((n,1), 1/n, dtype=torch.float32) #FIXME I am only for testing NOPROD
        # Get average of validator weights weighted by their stake?
        self.miner_gradients = []
        self.miner_weights = []
        self.miner_hotkeys = []
        for uid, hotkey in tqdm(enumerate(self.bittensor_network.metagraph.hotkeys)):
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
        self.miner_gradients = [
            gradients for gradients in self.miner_gradients if gradients is not None
        ]
        assert len(self.miner_gradients) > 0

        averaged_gradients = {
            name: torch.zeros_like(grad)
            for name, grad in self.miner_gradients[0].items()
        }

        for score, gradients in zip(
            self.validator_combined_weights, self.miner_gradients
        ):
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
            per_device_eval_batch_size=1,  # Dummy argument, necessary to specify but won't be used
            push_to_hub=True,  # Enable pushing to hub
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
                logging.info(
                    "Model updated from Hugging Face. Continuing training with new model..."
                )
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
    def __init__(
        self,
        model,
        local_dir,
        repo_id,
        hf_manager,
        chain_manager,
        bittensor_network,
        hf_token=os.environ.get("HF_TOKEN"),
    ):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.scored_gradients = None
        self.last_sync_time = 0
        self.bittensor_network = bittensor_network
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager

        # initialize mlflow
        if MLFLOW_ACTIVE:
                initialize_mlflow(
                    role="averager",
                    device="cuda",
                    version=VERSION,
                    mlflow_ui_url=MLFLOW_UI_URL,
                    current_model_name=CURRENT_MODEL_NAME)

    def average_gradients(self, beta=1.0):
        self.non_none_miner_gradients = [
            gradients for gradients in self.miner_gradients if gradients is not None
        ]
        assert len(self.non_none_miner_gradients) > 0
        averaged_gradients = {
            name: torch.zeros_like(grad)
            for name, grad in self.non_none_miner_gradients[0].items()
        }

        for score, gradients in zip(
            self.validator_combined_weights, self.miner_gradients
        ):
            logging.info("Averaging Gradient")
            if gradients is not None:
                for (name, grad), (param_name, param) in zip(
                    gradients.items(), self.model.named_parameters()
                ):
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
                logging.info(
                    "Model updated from Hugging Face. Continuing training with new model..."
                )
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
    def __init__(
        self,
        model,
        local_dir,
        hf_manager,
        chain_manager,
        bittensor_network=None,
        hf_token=os.environ.get("HF_TOKEN"),
    ):
        super().__init__(
            model,
            local_dir,
            hf_manager=hf_manager,
            chain_manager=chain_manager,
            repo_id=None,
            bittensor_network=bittensor_network,
            hf_token=hf_token,
        )
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
        logging.info(
            f"Model saved locally in {self.local_dir} instead of pushing to Hugging Face Hub."
        )

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.local_dir, exist_ok=True)
        model_save_path = os.path.join(self.local_dir, "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")


class ParameterizedAverager(DeltaAverager):
    # __init__(self, model, local_dir, repo_id,hf_manager, chain_manager,bittensor_network, hf_token=os.environ.get("HF_TOKEN"))
    def __init__(
        self,
        model,
        local_dir,
        gradients_dir,
        device,
        repo_id=None,
        hf_manager=None,
        chain_manager=None,
        bittensor_network=None,
        hf_token=os.environ.get("HF_TOKEN"),
        check_update_interval=300,
    ):
        DeltaAverager.__init__(
            self,
            model,
            local_dir=local_dir,
            repo_id=repo_id,
            hf_manager=hf_manager,
            chain_manager=chain_manager,
            bittensor_network=bittensor_network,
            hf_token=hf_token,
        )
        self.device = device
        self.last_pull_time = 0
        self.check_update_interval = check_update_interval
        self.gradients_dir = gradients_dir


    def get_model_paths(self, gradient_file_name="gradients.pt"):
        self.model_paths = []
        self.bittensor_network.sync(lite=False)
        validator_uids = self.bittensor_network.get_validator_uids(
            vpermit_tao_limit=1024
        )
        self.validator_combined_weights = torch.mean(
                self.bittensor_network.metagraph.W[validator_uids, :], axis=0
            )
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                # gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None and self.validator_combined_weights[uid] > 0:
                    self.model_paths.append(
                        {"repo_id": repo_id, "hotkey": hotkey, "uid": uid}
                    )
            except Exception as e:
                logging.debug(f"Receiving repo failed due to: {e}")

    def store_weight_delta(self, weight_delta, hotkey):
        """
        Save the weight_delta state_dict to a local directory at regular intervals.
        """
        os.makedirs(self.gradients_dir, exist_ok=True)
        save_path = os.path.join(self.gradients_dir, f"weight_delta_{hotkey}.pt")
        torch.save(weight_delta, save_path)

    def load_weight_delta(self, hotkey):
        """
        Load the cached weight_delta state_dict from the local directory.
        """
        load_path = os.path.join(self.gradients_dir, f"weight_delta_{hotkey}.pt")
        if os.path.exists(load_path):
            return torch.load(load_path, map_location=self.device)
        else:
            return None

    def cache_params_locally(self):
        self.num_models = 0
        self.active_hotkeys = []
        for model_path in self.model_paths:
            if model_path is None:
                continue
            else:
                #try:
                weight_delta = self.hf_manager.receive_gradients(model_path["repo_id"])
                false_model_flag = False
                try:
                    for name, param in weight_delta.items():
                        if weight_delta[name].shape != self.model.state_dict()[name].shape:
                            false_model_flag = True
                except AttributeError:
                    false_model_flag = True
                if false_model_flag:
                    continue
                self.store_weight_delta(weight_delta, model_path["hotkey"])
                if weight_delta is None:
                    raise ValueError(f"Failed to receive gradients at: {model_path['repo_id']}")
                self.num_models +=1
                self.active_hotkeys.append(model_path["hotkey"])
                #except Exception as e:
                #    logging.warning(f"Failed to get model at {model_path['hotkey']}: {e}")
        if len(self.active_hotkeys) > 0:
            assert len(self.active_hotkeys) == self.num_models
        # if time.time() - self.last_cache_time > self.caching_interval:

    def get_averaged_params(self):
        if self.weights is None:
            self.weights = torch.ones(
                    (self.num_models, len(list(self.model.parameters()))),
                    device=self.device,
                )
            for uid in len(range(self.validator_combined_weights)):
                self.weights[uid, :] = self.weights[uid, :] * self.validator_combined_weights[uid]

            normalized_weights = nn.functional.softmax(self.weights,dim=0)
        else:
            normalized_weights = nn.functional.softmax(self.weights,dim=0)

        averaged_gradients = {
            name: torch.zeros_like(grad) for name, grad in self.model.named_parameters()
        }
        
        for params, weight in zip(self.lazy_load_params(), normalized_weights):
            if params is None or torch.all(weight == 0):
                continue
            for j, (name_reconstructed_model, param_reconstructed_model) in enumerate(
                params.items()
            ):
                param_reconstructed_model = param_reconstructed_model.to(self.device)
                averaged_gradients[name_reconstructed_model] = averaged_gradients[name_reconstructed_model].to(self.device)
                try:
                    averaged_gradients[name_reconstructed_model] += (param_reconstructed_model * weight[j]) #FIXME make weights per param
                except Exception as e:
                    #logging.warning(f"Skipping parameter due to: {e}")
                    pass

        return averaged_gradients

    def check_cached_parameters(self):
        logging.info("Clearing NaNs")
        with torch.no_grad():
            for uid, hotkey in enumerate(self.active_hotkeys):
                weight_delta = self.load_weight_delta(hotkey)
                
                if self.have_nans(weight_delta):
                    self.active_hotkeys.pop(uid)
                    self.num_models -= 1

    @staticmethod
    def have_nans(aggregated_gradients):
        for tensor in aggregated_gradients.values():
            if torch.isnan(tensor).any():
                #logging.debug("NaN values detected in the aggregated gradients.")
                return True
        return False

    def lazy_load_params(self):
        base_model = torch.load(
                os.path.join(
                    self.hf_manager.get_local_model_directory(), "averaged_model.pt"
                ),
                map_location=self.device,
            ) 
        with torch.no_grad():
            for uid, hotkey in tqdm(enumerate(self.active_hotkeys)):
                weight_delta = self.load_weight_delta(hotkey)
                # if weight_delta is None:
                #     yield None
                #     continue
                # self.model.state_dict()
                if self.have_nans(weight_delta):
                    continue
                for name, delta_param in weight_delta.items():
                    weight_delta[name] = weight_delta[name].to(self.device)
                    base_model[name] = base_model[name].to(self.device)
                    try:
                        weight_delta[name] = weight_delta[name] + base_model[name]
                        
                    except Exception as e:
                        #logging.warning(f"Error loading param: {e}")
                        pass
                yield weight_delta

    def get_averaged_model(self):
        averaged_params = self.get_averaged_params()
        for param, averaged_param in zip(
            self.model.parameters(), averaged_params.values()
        ):
            param.data.copy_(averaged_param.data)
        self.model.to(self.device)
        return self.model

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.hf_manager.get_local_model_directory(), exist_ok=True)
        model_save_path = os.path.join(self.hf_manager.get_local_model_directory(), "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

    def meta_learning(self, val_loader, meta_epochs, lr, accumulation_steps=128):
        criterion = nn.CrossEntropyLoss()
        self.weights = None  # nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))
        for epoch in range(meta_epochs):
            for epoch in range(meta_epochs):
                total_loss = 0
                correct_predictions = 0
                total_samples = 0

                for batch_count, batch in enumerate(val_loader):
                    averaged_model = self.get_averaged_model()

                    outputs = averaged_model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
                    )
                    val_loss = outputs.loss
                    total_loss += val_loss.item() * batch["input_ids"].size(0)
                    total_samples += batch["input_ids"].size(0)

                    val_loss.backward()
                    if (batch_count + 1) % accumulation_steps == 0 or batch_count + 1 == len(val_loader):
                        with torch.no_grad():
                            softmax_weights = nn.functional.softmax(self.weights, dim=0)
                            grad_weights = torch.zeros_like(self.weights)

                            for i, (model, softmax_weights_i) in enumerate(zip(self.lazy_load_params(), softmax_weights)):
                                for j, (model_param, main_param) in enumerate(zip(model.values(), averaged_model.parameters())):
                                    if main_param.grad is not None:
                                        grad_contribution = main_param.grad * (model_param - main_param)
                                        grad_contribution_sum = torch.sum(grad_contribution)

                                        # Direct vectorized update to grad_weights for all models
                                        grad_weights[:, j] += grad_contribution_sum * (softmax_weights_i[j] - softmax_weights[:, j] * softmax_weights_i[j])

                                        # Handle the diagonal term separately
                                        grad_weights[i, j] += grad_contribution_sum * softmax_weights_i[j]
                            
                            logging.info(f"Gradient check: {main_param.grad}")
                            self.weights.data -= (lr / accumulation_steps * grad_weights)
                            for main_param in averaged_model.parameters():
                                main_param.grad.zero_()
                            
                            #grad_weights = torch.clamp(grad_weights,min=-1,max=1)
                            if (batch_count * epoch+1) % 1 == 0:
                                logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {val_loss.item():.4f}, Weights: {torch.mean(self.weights,dim=1)}")

                average_loss = total_loss / total_samples
                perplexity = math.exp(average_loss) 
                
                if MLFLOW_ACTIVE:
                    step = int(time.time())
                    log_model_metrics(step=step, loss_averaged = average_loss, perplexity_averaged = perplexity)

                logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {average_loss:.4f},Perplexity: {perplexity}, Weights: {torch.mean(self.weights,dim=1)}")
                
        return self.get_averaged_model()
    

    def run_periodic_averaging(self, val_loader, meta_epochs, lr, t, accumulation_steps=128):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if time.time() - self.last_pull_time >= self.check_update_interval:
                if self.hf_manager.check_for_new_submissions(
                    self.hf_manager.model_repo_id
                ):
                    logging.info(
                        "Averaged model updated on Hugging Face. Pulling latest model..."
                    )
                    self.hf_manager.pull_latest_model()
                    time.sleep(10)  # just to give enough time for pull
                    self.model = self.hf_manager.update_model(self.model)
                    self.model = self.model.to(self.device)
                    #optimizer = AdamW(
                    #    self.model.parameters(), lr=5e-5
                    #)  # Reinitialize the optimizer
                    self.base_weights = {
                        name: param.clone()
                        for name, param in self.model.named_parameters()
                    }

                self.last_pull_time = time.time()

            # self.get_model_paths()
            # self.num_models = len(self.model_paths)
            # self.cache_params_locally()
            self.bittensor_network.sync(lite=False)
            validator_uids = self.bittensor_network.get_validator_uids(
            vpermit_tao_limit=1024
            )
            self.validator_combined_weights = torch.mean(
                self.bittensor_network.metagraph.W[validator_uids, :], axis=0
            )
            filenames = os.listdir(self.gradients_dir)
            self.active_hotkeys = [filename.split('_')[2].split('.')[0] for filename in filenames if filename.startswith('weight_delta')]
            self.num_models = len(self.active_hotkeys)
            self.check_cached_parameters()
            logging.info(f"Total number of models: {self.num_models}")

            self.model = self.meta_learning(val_loader, meta_epochs, lr, accumulation_steps)
            # self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            #self.hf_manager.push_to_hf_hub(path_to_model= "averaged_model.pt")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")


class TopKAverager(DeltaAverager):
    # __init__(self, model, local_dir, repo_id,hf_manager, chain_manager,bittensor_network, hf_token=os.environ.get("HF_TOKEN"))
    def __init__(
        self,
        model,
        local_dir,
        gradients_dir,
        device,
        repo_id=None,
        hf_manager=None,
        chain_manager=None,
        bittensor_network=None,
        hf_token=os.environ.get("HF_TOKEN"),
        check_update_interval=300,
    ):
        DeltaAverager.__init__(
            self,
            model,
            local_dir=local_dir,
            repo_id=repo_id,
            hf_manager=hf_manager,
            chain_manager=chain_manager,
            bittensor_network=bittensor_network,
            hf_token=hf_token,
        )
        self.device = device
        self.last_pull_time = 0
        self.check_update_interval = check_update_interval
        self.gradients_dir = gradients_dir


    def get_model_paths(self, gradient_file_name="gradients.pt"):
        self.model_paths = []
        # self.bittensor_network.sync(lite=False)
        # validator_uids = self.bittensor_network.get_validator_uids(
        #     vpermit_tao_limit=1024
        # )
        # self.validator_combined_weights = torch.mean(
        #         self.bittensor_network.metagraph.W[validator_uids, :], axis=0
        #     )
        for uid, hotkey in tqdm(enumerate(self.bittensor_network.metagraph.hotkeys)):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                # gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None and self.validator_combined_weights[uid] > 0:
                    self.model_paths.append(
                        {"repo_id": repo_id, "hotkey": hotkey, "uid": uid, "weight":self.validator_combined_weights[uid] }
                    )
            except Exception as e:
                logging.debug(f"Receiving repo failed due to: {e}")

    def store_weight_delta(self, weight_delta, hotkey):
        """
        Save the weight_delta state_dict to a local directory at regular intervals.
        """
        os.makedirs(self.gradients_dir, exist_ok=True)
        save_path = os.path.join(self.gradients_dir, f"weight_delta_{hotkey}.pt")
        torch.save(weight_delta, save_path)

    def load_weight_delta(self, hotkey):
        """
        Load the cached weight_delta state_dict from the local directory.
        """
        load_path = os.path.join(self.gradients_dir, f"weight_delta_{hotkey}.pt")
        if os.path.exists(load_path):
            return torch.load(load_path, map_location=self.device)
        else:
            return None

    
    def filter_hotkeys_for_nans(self):
        new_active_hotkeys = []
        for hotkey in self.active_hotkeys:
            weight_delta = self.load_weight_delta(hotkey)
            if not self.have_nans(weight_delta):
                new_active_hotkeys.append(hotkey)
            else:
                logging.info(f"Model with hotkey {hotkey} excluded due to NaNs.")
        self.active_hotkeys = new_active_hotkeys
        self.num_models = len(self.active_hotkeys)
        assert self.num_models > 0

    def get_top_k_models(self, top_k):
        # Assuming self.model_paths includes model details and associated weights
        # Sort model paths by weight
        sorted_model_paths = sorted(self.model_paths, key=lambda x: x['weight'], reverse=True)
        self.top_k_model_paths = sorted_model_paths[:top_k]
        
        # Update active hotkeys to only include those from the top-k models
        self.active_validator_weights = torch.tensor([model['weight'] for model in self.top_k_model_paths])
        self.active_hotkeys = [model['hotkey'] for model in self.top_k_model_paths]
        self.num_models = len(self.active_hotkeys)



    def cache_params_locally(self):
        for model_path in self.top_k_model_paths:
            if model_path is None:
                continue
            else:
                #try:
                weight_delta = self.hf_manager.receive_gradients(model_path["repo_id"])
                false_model_flag = False
                try:
                    for name, param in weight_delta.items():
                        if weight_delta[name].shape != self.model.state_dict()[name].shape:
                            false_model_flag = True
                except AttributeError:
                    false_model_flag = True
                if false_model_flag:
                    continue
                self.store_weight_delta(weight_delta, model_path["hotkey"])
                if weight_delta is None:
                    raise ValueError(f"Failed to receive gradients at: {model_path['repo_id']}")
                #self.num_models +=1
                #self.active_hotkeys.append(model_path["hotkey"])
                #except Exception as e:
                #    logging.warning(f"Failed to get model at {model_path['hotkey']}: {e}")
        if len(self.active_hotkeys) > 0:
            assert len(self.active_hotkeys) == self.num_models
        # if time.time() - self.last_cache_time > self.caching_interval:



    def get_averaged_params(self):
        if self.weights is None:
            self.weights = torch.ones(
                    (self.num_models, len(list(self.model.parameters()))),
                    device=self.device,
                )

            for uid in range(self.active_validator_weights.shape[0]):
                self.weights[uid, :] = self.weights[uid, :] * self.validator_combined_weights[uid]

            normalized_weights = nn.functional.softmax(self.weights,dim=0)
        else:
            normalized_weights = nn.functional.softmax(self.weights,dim=0)

        averaged_gradients = {
            name: torch.zeros_like(grad) for name, grad in self.model.named_parameters()
        }
        
        for params, weight in zip(self.lazy_load_params(), normalized_weights):
            if params is None or torch.all(weight == 0):
                continue
            for j, (name_reconstructed_model, param_reconstructed_model) in enumerate(
                params.items()
            ):
                #param_reconstructed_model = param_reconstructed_model.to(self.device)
                averaged_gradients[name_reconstructed_model] = averaged_gradients[name_reconstructed_model].to(self.device)
                try:
                    averaged_gradients[name_reconstructed_model] += (param_reconstructed_model * weight[j]) #FIXME make weights per param
                except Exception as e:
                    #logging.warning(f"Skipping parameter due to: {e}")
                    pass

        return averaged_gradients


    @staticmethod
    def have_nans(aggregated_gradients):
        for tensor in aggregated_gradients.values():
            if torch.isnan(tensor).any():
                #logging.debug("NaN values detected in the aggregated gradients.")
                return True
        return False

    def lazy_load_params(self):
        
        with torch.no_grad():
            for uid, hotkey in tqdm(enumerate(self.active_hotkeys)):
                weight_delta = self.load_weight_delta(hotkey)
                # if weight_delta is None:
                #     yield None
                #     continue
                # self.model.state_dict()
                if self.have_nans(weight_delta):
                    continue
                for name, delta_param in weight_delta.items():
                    weight_delta[name] = weight_delta[name].to(self.device)
                    # base_model[name] = base_model[name].to(self.device)
                    # try:
                    #     weight_delta[name] = weight_delta[name] + base_model[name]
                        
                    # except Exception as e:
                    #     #logging.warning(f"Error loading param: {e}")
                    #     pass
                yield weight_delta

    def get_averaged_model(self):
        base_model = torch.load(
                os.path.join(
                    self.hf_manager.get_local_model_directory(), "averaged_model.pt"
                ),
                map_location=self.device,
            ) 
        averaged_params = self.get_averaged_params()

        for param, averaged_param in zip(
            base_model, averaged_params.values()
        ):
            try:
                averaged_param[name] = averaged_param[name] + base_model[name]
                
            except Exception as e:
                #logging.warning(f"Error loading param: {e}")
                pass

        
        for param, averaged_param in zip(
            self.model.parameters(), averaged_params.values()
        ):
            param.data.copy_(averaged_param.data)
        self.model.to(self.device)
        return self.model

    def save_model(self, model):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.hf_manager.get_local_model_directory(), exist_ok=True)
        model_save_path = os.path.join(self.hf_manager.get_local_model_directory(), "averaged_model.pt")
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

    def meta_learning(self, val_loader, meta_epochs, lr, accumulation_steps=128):
        criterion = nn.CrossEntropyLoss()
        self.weights = None  # nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))
        best_model = None
        best_loss = float('inf')

        for epoch in range(meta_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for batch_count, batch in enumerate(val_loader):
                averaged_model = self.get_averaged_model()

                outputs = averaged_model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                val_loss = outputs.loss
                total_loss += val_loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                val_loss.backward()
                if (batch_count + 1) % accumulation_steps == 0 or batch_count + 1 == len(val_loader):
                    with torch.no_grad():
                        softmax_weights = nn.functional.softmax(self.weights, dim=0)
                        grad_weights = torch.zeros_like(self.weights)

                        for i, (model, softmax_weights_i) in enumerate(zip(self.lazy_load_params(), softmax_weights)):
                            for j, (model_param, main_param) in enumerate(zip(model.values(), averaged_model.parameters())):
                                if main_param.grad is not None:
                                    grad_contribution = main_param.grad * (model_param - main_param)
                                    grad_contribution_sum = torch.sum(grad_contribution)

                                    # Direct vectorized update to grad_weights for all models
                                    grad_weights[:, j] += grad_contribution_sum * (softmax_weights_i[j] - softmax_weights[:, j] * softmax_weights_i[j])

                                    # Handle the diagonal term separately
                                    grad_weights[i, j] += grad_contribution_sum * softmax_weights_i[j]
                        
                        #logging.info(f"Gradient check: {main_param.grad}")
                        self.weights.data -= (lr / accumulation_steps * grad_weights)
                        for main_param in averaged_model.parameters():
                            main_param.grad.zero_()
                        
                        #grad_weights = torch.clamp(grad_weights,min=-1,max=1)
                        #if (batch_count * epoch+1) % 1 == 0:
                        #    logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {val_loss.item():.4f}, Weights: {torch.mean(self.weights,dim=1)}")
            try:
                average_loss = total_loss / total_samples
                perplexity = math.exp(average_loss) 
            except:
                average_loss = 9999
                perplexity = 99999999

            if average_loss < best_loss:
                best_loss = average_loss
                best_model = copy.deepcopy(averaged_model.state_dict())  
            
            if MLFLOW_ACTIVE:
                #step = int(time.time())
                #log_model_metrics(step=step, loss_averaged = average_loss, perplexity_averaged = perplexity)
                mlflow.log_metric("loss", average_loss, step=time.time() )
                mlflow.log_metric("perplexity", average_perplexity, step=time.time() )

            logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {average_loss:.4f},Perplexity: {perplexity}, Weights: {torch.mean(self.weights,dim=1)}")
        
        if best_model is not None:
            self.model.load_state_dict(best_model)
        return self.model
    

    def run_periodic_averaging(self, val_loader, meta_epochs, lr, t, accumulation_steps=128):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if time.time() - self.last_pull_time >= self.check_update_interval:
                if self.hf_manager.check_for_new_submissions(
                    self.hf_manager.model_repo_id
                ):
                    logging.info(
                        "Averaged model updated on Hugging Face. Pulling latest model..."
                    )
                    self.hf_manager.pull_latest_model()
                    time.sleep(10)  # just to give enough time for pull
                    self.model = self.hf_manager.update_model(self.model)
                    self.model = self.model.to(self.device)
                    #optimizer = AdamW(
                    #    self.model.parameters(), lr=5e-5
                    #)  # Reinitialize the optimizer
                    self.base_weights = {
                        name: param.clone()
                        for name, param in self.model.named_parameters()
                    }

                self.last_pull_time = time.time()
            
            self.bittensor_network.sync(lite=False)
            validator_uids = self.bittensor_network.get_validator_uids(
            vpermit_tao_limit=1024
            )
            self.validator_combined_weights = torch.mean(
                self.bittensor_network.metagraph.W[validator_uids, :], axis=0
            )
            self.get_model_paths()   

            self.get_top_k_models(5)
            self.cache_params_locally()
            self.filter_hotkeys_for_nans()
            
            #filenames = os.listdir(self.gradients_dir)
            #self.active_hotkeys = [filename.split('_')[2].split('.')[0] for filename in filenames if filename.startswith('weight_delta')]
            #self.num_models = len(self.active_hotkeys)
            
            logging.info(f"Total number of models: {self.num_models}")

            model = self.meta_learning(val_loader, meta_epochs, lr, accumulation_steps)
            # self.apply_averaged_gradients(averaged_weights)
            self.save_model(model)
            #self.hf_manager.push_to_hf_hub(path_to_model= "averaged_model.pt")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")


class LocalParameterizedAverager(LocalAverager):
    def __init__(
        self,
        model,
        local_dir,
        device,
        hf_manager,
        chain_manager=None,
        bittensor_network=None,
        hf_token=os.environ.get("HF_TOKEN"),
    ):
        LocalAverager.__init__(
            self,
            model,
            local_dir,
            hf_manager=hf_manager,
            chain_manager=chain_manager,
            bittensor_network=bittensor_network,
            hf_token=hf_token,
        )
        self.device = device

    def get_model_paths(self, gradient_file_name="gradients.pt"):
        self.model_paths = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            try:
                repo_id = self.chain_manager.retrieve_hf_repo(hotkey)
                # gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None:
                    self.model_paths.append(repo_id)
            except Exception as e:
                logging.debug(f"Receiving gradients failed due to: {e}")

    def get_averaged_params(self):
        if self.weights is None:
            self.weights = nn.functional.softmax(
                torch.ones(
                    (self.num_models, len(list(self.model.parameters()))),
                    device=self.device,
                ),
                dim=0,
            )
        averaged_gradients = {
            name: torch.zeros_like(grad) for name, grad in self.model.named_parameters()
        }
        for params, weight in zip(self.lazy_load_params(), self.weights):
            for j, (name_reconstructed_model, param_reconstructed_model) in enumerate(
                params.items()
            ):
                averaged_gradients[name_reconstructed_model] += (
                    param_reconstructed_model * weight[j]
                )  # FIXME make weights per param

        return averaged_gradients

    def lazy_load_params(self):
        for model_path in self.model_paths:
            if model_path is None:
                yield None
            else:
                weight_delta = torch.load(
                    os.path.join(model_path, "gradients.pt"), map_location=self.device
                )
                base_model = torch.load(
                    os.path.join(self.local_dir, "averaged_model.pt"),
                    map_location=self.device,
                )  # self.model.state_dict()
                for name, delta_param in weight_delta.items():
                    weight_delta[name] = weight_delta[name].to(self.device)
                    base_model[name] = base_model[name].to(self.device)
                    weight_delta[name] = weight_delta[name] + base_model[name]
                yield weight_delta

    def get_averaged_model(self):
        averaged_params = self.get_averaged_params()
        for param, averaged_param in zip(
            self.model.parameters(), averaged_params.values()
        ):
            param.data.copy_(averaged_param.data)
        self.model.to(self.device)
        return self.model

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.local_dir, exist_ok=True)
        model_save_path = os.path.join(self.local_dir, "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

    def meta_learning(self, val_loader, meta_epochs, lr):
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD([self.weights], lr=lr)
        self.weights = None  # nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))
        for epoch in range(meta_epochs):
            # Outer loop: Update averaging weights

            # val_loss = evaluate_model(averaged_model, val_loader, criterion, self.device)

            # self.model.eval()

            for epoch in range(meta_epochs):
                total_loss = 0
                correct_predictions = 0
                total_samples = 0

                for batch_count, batch in enumerate(val_loader):
                    averaged_model = self.get_averaged_model()

                    images, labels = batch
                    outputs = averaged_model(images)
                    val_loss = F.cross_entropy(outputs, labels)
                    total_loss += val_loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                    val_loss.backward()
                    with torch.no_grad():
                        grad_weights = torch.zeros_like(self.weights)

                        # for main_param in averaged_model.parameters():
                        #     if main_param.grad is not None:
                        #         main_param.grad = torch.clamp(main_param.grad,min=-0.1,max=0.1)

                        for i, model in enumerate(self.lazy_load_params()):
                            for j, (model_param, main_param) in enumerate(
                                zip(model.values(), averaged_model.parameters())
                            ):
                                if main_param.grad is not None:
                                    grad_weights[i, j] += torch.sum(
                                        main_param.grad * (model_param - main_param)
                                    )

                        for main_param in averaged_model.parameters():
                            main_param.grad.zero_()

                        # grad_weights = torch.clamp(grad_weights,min=-1,max=1)
                        self.weights.data -= lr * grad_weights
                        # if (batch_count * epoch+1) % 100:
                        #    logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {val_loss.item():.4f}, Weights: {self.weights}")

                average_loss = total_loss / total_samples
                accuracy = correct_predictions / total_samples
                logging.info(
                    f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {average_loss:.4f},Accuracy: {accuracy}, Weights: {self.weights}"
                )

        return self.get_averaged_model()

    def run_periodic_averaging(self, val_loader, meta_epochs, lr, t):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if self.hf_manager.check_for_new_submissions():
                logging.info(
                    "Model updated from Hugging Face. Continuing training with new model..."
                )
                self.model = self.hf_manager.update_model(self.model)

            self.get_model_paths()
            self.num_models = len(self.model_paths)

            self.model = self.meta_learning(val_loader, meta_epochs, lr)
            # self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")


class LocalLLMParameterizedAverager(LocalParameterizedAverager):
    def meta_learning(self, val_loader, meta_epochs, lr):
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD([self.weights], lr=lr)
        self.weights = None  # nn.Parameter(nn.functional.softmax(torch.ones(self.num_models, device=self.device), dim=0))
        for epoch in range(meta_epochs):
            # Outer loop: Update averaging weights

            # val_loss = evaluate_model(averaged_model, val_loader, criterion, self.device)

            # self.model.eval()

            for epoch in range(meta_epochs):
                total_loss = 0
                total_samples = 0

                for batch_num, batch in enumerate(
                    val_loader
                ):  # FIXME turn me into a generator?
                    averaged_model = self.get_averaged_model()
                    averaged_model.train()
                    optimizer = optim.SGD(
                        averaged_model.parameters(), lr=1010
                    )  # This is only used to clear grads
                    outputs = averaged_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    val_loss = outputs.loss
                    total_loss += val_loss.item() * batch["input_ids"].size(0)
                    total_samples += batch["input_ids"].size(0)

                    val_loss.backward()
                    with torch.no_grad():
                        grad_weights = torch.zeros_like(self.weights)

                        for i, model in enumerate(self.lazy_load_params()):
                            for j, (model_param, main_param) in enumerate(
                                zip(model.values(), averaged_model.parameters())
                            ):
                                if main_param.grad is not None:
                                    grad_weights[i, j] += torch.sum(
                                        main_param.grad * (model_param - main_param)
                                    )

                        for main_param in averaged_model.parameters():
                            main_param.grad.zero_()
                        optimizer.zero_grad()
                        # grad_weights = torch.clamp(grad_weights,min=-1,max=1)
                        self.weights.data -= lr * grad_weights
                        if (batch_num * epoch + 1) % 100:
                            average_loss = total_loss / total_samples
                            # logging.info(f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {average_loss}, Weights: {self.weights}")

                average_loss = total_loss / total_samples
                try:
                    perplexity = math.exp(average_loss)
                except:
                    perplexity = 999999
                logging.info(
                    f"Meta-Epoch [{epoch+1}/{meta_epochs}], Validation Loss: {average_loss:.4f}, Perplexity: {perplexity}, Weights: {self.weights}"
                )

        return self.get_averaged_model()


class GeneticAverager(nn.Module):
    def __init__(
        self,
        model,
        local_dir,
        device,
        hf_manager,
        chain_manager=None,
        bittensor_network=None,
        hf_token=os.environ.get("HF_TOKEN"),
    ):
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
                # gradient = self.receive_gradients(repo_id=repo_id)
                if repo_id is not None:
                    self.model_paths.append(repo_id)
            except Exception as e:
                logging.debug(f"Receiving gradients failed due to: {e}")

    def lazy_load_params(self):
        for model_path in self.model_paths:
            if model_path is None:
                yield None
            else:
                weight_delta = torch.load(
                    os.path.join(model_path, "gradients.pt"), map_location="cpu"
                )
                yield weight_delta

    def get_averaged_params(self, weights):
        averaged_params = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
        }
        for params, weight in zip(self.lazy_load_params(), weights):
            for (name_weight_delta, param_weight_delta), (
                name_base_model,
                param_base_model,
            ) in zip(params.items(), self.model.named_parameters()):
                averaged_params[name_base_model] += (
                    param_weight_delta + param_base_model
                ) * weight
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
            fitness_scores.append(
                -average_loss
            )  # Negative because lower loss is better
        return fitness_scores

    def evolve_population(self, population, val_loader):
        fitness_scores = self.evaluate_population(val_loader, population)
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order of fitness
        best_individuals = [
            population[i] for i in sorted_indices[: len(population) // 2]
        ]  # Select top 50%

        # Reproduce with mutation
        new_population = []
        while len(new_population) < len(population):
            parent = random.choice(best_individuals)
            child = parent + torch.randn_like(parent) * self.sigma  # Gaussian mutation
            new_population.append(child)
        return new_population

    def run_evolution(self, val_loader):
        # Initialize population
        population = [
            torch.rand(len(self.model_paths), device=self.device)
            for _ in range(self.population_size)
        ]

        for generation in tqdm(range(self.num_generations)):
            population = self.evolve_population(population, val_loader)
            best_weights = population[0]
            best_fitness = self.evaluate_population(val_loader, [best_weights])[0]
            print(f"Generation {generation}, Best Fitness: {best_fitness}")

        return self.get_averaged_model(best_weights)

    def run_periodic_averaging(self, val_loader, t=40):
        while True:
            logging.info("Averaging Beginning")
            start_time = time.time()

            if self.hf_manager.check_for_new_submissions():
                logging.info(
                    "Model updated from Hugging Face. Continuing training with new model..."
                )
                self.model = self.hf_manager.update_model(self.model)

            self.get_model_paths()
            self.num_models = len(self.model_paths)
            self.weights = nn.Parameter(
                nn.functional.softmax(
                    torch.ones(self.num_models, device=self.device), dim=0
                )
            )

            self.model = self.run_evolution(val_loader)
            # self.apply_averaged_gradients(averaged_weights)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")

            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)

            logging.info("Averaging Done")
