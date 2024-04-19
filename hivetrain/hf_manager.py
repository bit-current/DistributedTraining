import os
import torch
import hashlib
<<<<<<< Updated upstream

=======
from dotenv import load_dotenv
>>>>>>> Stashed changes
from huggingface_hub import HfApi, Repository
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFManager:
    def __init__(self, my_repo_id, averaged_model_repo_id, hf_token, local_dir='.', model_dir = None):
        self.my_repo_id = my_repo_id
        self.model_repo_id = averaged_model_repo_id
        self.hf_token = hf_token

       
        self.gradient_repo = Repository(local_weights_dir=os.path.join(local_dir, my_repo_id.split('/')[-1]), clone_from=my_repo_id, use_auth_token=hf_token)
        self.local_gradient_dir = os.path.join(local_dir, my_repo_id.split('/')[-1])
        
        self.model_dir = model_dir if model_dir else os.path.join(local_dir, averaged_model_repo_id.split('/')[-1])
        self.model_repo = Repository(local_weights_dir=self.model_dir, clone_from=averaged_model_repo_id, use_auth_token=hf_token)

        self.api = HfApi()
        self.latest_model_commit_sha = self.get_latest_commit_sha(self.model_repo)

    
    def get_latest_commit_sha(self, repo):
        """Fetches the latest commit SHA of the repository."""
        try:
            repo_info = self.api.repo_info(repo.local_dir.split('/')[-1], token=self.token)
            latest_commit_sha = repo_info.sha
            print(latest_commit_sha)
            return latest_commit_sha
        except Exception as e:
            print(f"Failed to fetch latest commit SHA: {e}")
            return None

    def check_for_new_submissions(self, repo):
        """Checks if there's a new submission in the repository."""
        current_commit_sha = self.get_latest_commit_sha(repo)
        if current_commit_sha != self.last_known_commit_sha:
            print("New submission found. Updating...")
            self.last_known_commit_sha = current_commit_sha
            return True
        return False

    # def update_model(self, model, model_path):
    #     """Updates an existing model's state dict from a .pt file."""
    #     model_file_path = os.path.join(self.local_weights_dir, model_path)
    #     if os.path.exists(model_file_path):
    #         model_state_dict = torch.load(model_file_path)
    #         model.load_state_dict(model_state_dict)
    #         model.train()  # Or model.eval(), depending on your use case
    #         print(f"Model updated from local path: {model_file_path}")
    #     else:
    #         print(f"Model file not found: {model_file_path}")
    

    def update_model(self, model, model_file_name='averaged_model.pt'):
        """Updates an existing model's state dict from a .pt file."""
        model_path = os.path.join(self.model_dir, model_file_name)
        if os.path.exists(model_path):
            model_state_dict = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            model.train()
            print(f"Model updated from local path: {model_path}")
            return model_state_dict
        else:
            raise FileNotFoundError(f"{model_file_name} not found in the repository.")
    
    def get_local_gradient_directory(self):
        """Return the local directory of the repository."""
        return self.local_gradient_dir
    
    def get_local_model_directory(self):
        """Return the local directory of the repository."""
        return self.model_dir
    

<<<<<<< Updated upstream
=======
    def pull_latest_model(self):
        self.model_repo.git_pull()
>>>>>>> Stashed changes

class LocalHFManager:
    def __init__(self, my_repo_id="local_models"):
        self.my_repo_id = my_repo_id
        # Ensure the local directory exists
        os.makedirs(self.my_repo_id, exist_ok=True)
        self.model_hash_file = os.path.join(self.my_repo_id, "model_hash.txt")
        # Initialize model hash value
        self.last_known_hash = None

    def set_model_hash(self, hash_value):
        """Sets and saves the latest model hash to the hash file."""
        with open(self.model_hash_file, "w") as file:
            file.write(hash_value)
        print(f"Set latest model hash to: {hash_value}")

    def check_for_new_submissions(self):
        """Checks if a new or updated model is available."""
        model_file_path = os.path.join(self.my_repo_id, "averaged_model.pt")
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
        model_file_path = os.path.join(self.my_repo_id, "averaged_model.pt")
        if os.path.exists(model_file_path):
            model_state_dict = torch.load(model_file_path)
            model.load_state_dict(model_state_dict)
            model.train()  # Or model.eval(), depending on your use case
            return model
            print(f"Model updated from local path: {model_file_path}")
        else:
            print(f"Model file not found: {model_file_path}")
