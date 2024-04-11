from huggingface_hub import HfApi, Repository
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFManager:
    def __init__(self, repo_id="mekaneeky/test_me"):
        self.repo_id = repo_id
        self.last_known_commit_sha = self.get_latest_commit_sha()

    def get_latest_commit_sha(self):
        """Fetches the latest commit SHA of the repository."""
        api = HfApi()
        repo_info = api.repo_info(self.repo_id)
        latest_commit_sha = repo_info.sha
        return latest_commit_sha

    def check_for_new_submissions(self):
        """Checks if there's a new submission in the repository."""
        current_commit_sha = self.get_latest_commit_sha()
        if current_commit_sha != self.last_known_commit_sha:
            print("New submission found. Updating model...")
            self.last_known_commit_sha = current_commit_sha
            return True
        return False

    def update_model(self, model, model_path):
        """Updates an existing model's state dict from a .pt file."""
        model_file_path = os.path.join(self.local_dir, model_path)
        if os.path.exists(model_file_path):
            model_state_dict = torch.load(model_file_path)
            model.load_state_dict(model_state_dict)
            model.train()  # Or model.eval(), depending on your use case
            print(f"Model updated from local path: {model_file_path}")
        else:
            print(f"Model file not found: {model_file_path}")


import os
import torch

import os
import torch

import os
import hashlib

class LocalHFManager:
    def __init__(self, repo_id="local_models"):
        self.repo_id = repo_id
        # Ensure the local directory exists
        os.makedirs(self.repo_id, exist_ok=True)
        self.model_hash_file = os.path.join(self.repo_id, "model_hash.txt")
        # Initialize model hash value
        self.last_known_hash = None

    def set_model_hash(self, hash_value):
        """Sets and saves the latest model hash to the hash file."""
        with open(self.model_hash_file, "w") as file:
            file.write(hash_value)
        print(f"Set latest model hash to: {hash_value}")

    def check_for_new_submissions(self):
        """Checks if a new or updated model is available."""
        model_file_path = os.path.join(self.repo_id, "averaged_model.pt")
        if not os.path.exists(model_file_path):
            print("No model available.")
            return False

        with open(model_file_path, "rb") as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()
            
        if self.last_known_hash is None or self.last_known_hash != file_hash:
            print("New or updated model found. Updating model...")
            self.last_known_hash = file_hash
            return True
        return False

    def update_model(self, model):
        """Updates an existing model's state dict from a .pt file."""
        model_file_path = os.path.join(self.repo_id, "averaged_model.pt")
        if os.path.exists(model_file_path):
            model_state_dict = torch.load(model_file_path)
            model.load_state_dict(model_state_dict)
            model.train()  # Or model.eval(), depending on your use case
            return model
            print(f"Model updated from local path: {model_file_path}")
        else:
            print(f"Model file not found: {model_file_path}")
