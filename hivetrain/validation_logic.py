import os
import hashlib
import torch
import time
import math
import mlflow
import mlflow.pytorch
from bittensor import logging
import logging
from copy import deepcopy
from hivetrain.btt_connector import BittensorNetwork
from hivetrain.config import Configurator
from hivetrain.config.mlflow_config import (
    MLFLOW_UI_URL,
    CURRENT_MODEL_NAME,
    MLFLOW_ACTIVE,
)
from hivetrain.utils.mlflow_utils import initialize_mlflow, log_model_metrics, VERSION
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F


args = Configurator.combine_configs()
BittensorNetwork.initialize(args)
MY_HOTKEY = BittensorNetwork.wallet.hotkey.ss58_address


class ModelValidator:
    def __init__(
        self,
        device,
        model,
        optimizer,
        data_loader,
        check_update_interval=300,
        bittensor_network=None,
        chain_manager=None,
        hf_manager=None,
        interval=300,
    ):
        self.device = device
        self.model = model
        self.model = self.model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.interval = interval  # Validation interval in seconds
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.bittensor_network = bittensor_network
        self.scores = {
            hotkey: 0.0 for hotkey in self.bittensor_network.metagraph.hotkeys
        }
        self.chain_manager = chain_manager
        self.hf_manager = hf_manager
        self.last_pull_time = 0
        self.check_update_interval = check_update_interval

        if MLFLOW_ACTIVE:
            initialize_mlflow(
                role="validator",
                device=self.device,
                version=VERSION,
                mlflow_ui_url=MLFLOW_UI_URL,
                current_model_name=CURRENT_MODEL_NAME,
                my_hotkey=MY_HOTKEY,
                check_update_interval=self.check_update_interval,
            )

    def update_model_weights(self, gradients, alpha=5e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= gradients[name] * alpha

    def evaluate_model(self, metric="perplexity"):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(
                self.data_loader
            ):  # FIXME turn me into a generator?
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

        average_loss = total_loss / total_samples
        perplexity = math.exp(average_loss) if metric == "perplexity" else None
        return average_loss, perplexity

    def validate_and_score(self):
        """Check if the model is changed on HF , Check if HF commit hash is updated?
        If true pull"""

        logging.info("!Receiving Gradients from chain")
        self.bittensor_network.sync(lite=True)  # FIXME too prone to issues

        if time.time() - self.last_pull_time >= self.check_update_interval:
            if self.hf_manager.check_for_new_submissions(self.hf_manager.model_repo_id):
                logging.info(
                    "Averaged model updated on Hugging Face. Pulling latest model..."
                )
                self.hf_manager.pull_latest_model()
                time.sleep(10)  # Give enough time for pull
                self.model = self.hf_manager.update_model(self.model)
                self.model = self.model.to(self.device)
                self.optimizer = AdamW(
                    self.model.parameters(), lr=5e-5
                )  # Reinitialize the optimizer
                self.base_weights = {
                    name: param.clone() for name, param in self.model.named_parameters()
                }
            self.last_pull_time = time.time()

        self.original_state_dict = deepcopy(self.model.state_dict())

        for uid, hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):
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

                if MLFLOW_ACTIVE:
                    metrics = {
                        f"loss_{hotkey_address}": loss.item(),
                        f"perplexity_{hotkey_address}": perplexity,
                        f"loss_score_{hotkey_address}": loss_score,
                        f"perplexity_score_{hotkey_address}": perplexity_score,
                    }

                    # Log metrics with dynamic names
                    log_model_metrics(step=int(current_time), **metrics)

            else:
                loss = 99999999.0
                perplexity = 99999999.0
                loss_score = 0.0
                perplexity_score = 0.0

                current_time = int(time.time())
                if MLFLOW_ACTIVE:
                    metrics = {
                        f"loss_{hotkey_address}": loss,
                        f"perplexity_{hotkey_address}": perplexity,
                        f"loss_score_{hotkey_address}": loss_score,
                        f"perplexity_score_{hotkey_address}": perplexity_score,
                    }
                    log_model_metrics(step=int(current_time), **metrics)

            self.scores[hotkey_address] = perplexity_score
            # log validator performance

            if uid == 1:
                if MLFLOW_ACTIVE:
                    try:
                        mlflow.log_param("Version of Code", VERSION)
                    except Exception as e:
                        return None

            # Reset the model to its original state
            logging.info(f"Loss: {loss}, Perplexity: {perplexity}")
            logging.info(
                f"Loss Score: {loss_score}, Perplexity Score: {perplexity_score}"
            )
            time.sleep(0.1)

        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.scores)

    def start_periodic_validation(self):
        while True:
            self.validate_and_score()
            logging.info(f"One round done sleeping for: {self.interval}")
            time.sleep(self.interval)

    def calculate_model_hash(self):
        model_hash = hashlib.sha256()
        for name, param in self.model.named_parameters():
            model_hash.update(name.encode("utf-8"))
            model_hash.update(param.data.cpu().numpy().tobytes())
        return model_hash.hexdigest()


class LocalValidator(ModelValidator):
    def __init__(
        self,
        model,
        optimizer,
        data_loader,
        bittensor_network=None,
        chain_manager=None,
        hf_manager=None,
        interval=3600,
        local_gradient_dir="local_gradients",
    ):
        super().__init__(
            model,
            optimizer,
            data_loader,
            bittensor_network,
            chain_manager,
            hf_manager,
            interval,
        )
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
                    try:
                        param.data = weight_deltas[name] + param.data
                    except Exception as e:
                        logging.warning(f"Error loading gradients: {e}")
    
class LocalDeltaValidator(DeltaValidator, LocalValidator):
    pass


class MNISTValidator(LocalValidator):
    def __init__(
        self,
        model,
        optimizer,
        data_loader,
        bittensor_network=None,
        chain_manager=None,
        hf_manager=None,
        interval=300,
        local_gradient_dir="local_gradients",
    ):
        super().__init__(
            model,
            optimizer,
            data_loader,
            bittensor_network,
            chain_manager,
            hf_manager,
            interval,
        )
        (
            self.base_loss,
            self.base_accuracy,
        ) = self.evaluate_model()  # Redefine to use accuracy for MNIST

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
