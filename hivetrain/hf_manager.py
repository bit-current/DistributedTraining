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

    def update_model(self, model_name):
        """Updates the model from the Hugging Face repository."""
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.train()
        return model


import os
import torch

class LocalHFManager:
    def __init__(self, repo_id="local_models"):
        self.local_dir = repo_id
        self.model_version_file = os.path.join(self.local_dir, "latest_version.txt")
        self.last_known_version = self.get_latest_version()

    def get_latest_version(self):
        """Fetches the latest model version from a local file."""
        try:
            with open(self.model_version_file, "r") as file:
                latest_version = file.read().strip()
            return latest_version
        except FileNotFoundError:
            print("Version file not found. Please ensure the version file exists.")
            return None

    def check_for_new_submissions(self):
        """Checks if there's a new model version available locally."""
        current_version = self.get_latest_version()
        if current_version != self.last_known_version:
            print("New model version found. Updating model...")
            self.last_known_version = current_version
            return True
        return False

    def update_model(self, model_path):
        """Loads a new model version from a local directory."""
        model_file_path = os.path.join(self.local_dir, model_path)
        if os.path.exists(model_file_path):
            model = torch.load(model_file_path)
            model.train()
            print(f"Model updated from local path: {model_file_path}")
            return model
        else:
            print(f"Model file not found: {model_file_path}")
            return None