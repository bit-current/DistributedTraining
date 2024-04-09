import time
from transformers import AdamW #FIXME replace me with LAMB
from huggingface_hub import Repository
import torch

class TrainingLoop:
    def __init__(self, model_name, data_loader, learning_rate=5e-5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()

        self.data_loader = data_loader
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def store_gradients(aggregated_gradients, hf_repo_path="your_hf_repo_path", gradient_file_name="gradients.pt", use_auth_token=True):
        # Save gradients to a file temporarily
        torch.save(aggregated_gradients, gradient_file_name)

        # Initialize repository
        repo = Repository(local_dir=hf_repo_path, use_auth_token=use_auth_token)
        
        # Move the gradient file to the repository and commit
        repo.git_add(gradient_file_name)
        repo.git_commit("Update gradients")
        
        # Push the changes to the Hugging Face repository
        repo.git_push()

    def train(self, epochs, hf_manager):
        last_send_time = time.time()
        self.optimizer.zero_grad()
        for epoch in range(epochs):
            # Check for new submissions at the start of each epoch
            if hf_manager.check_for_new_submissions():
                print("Model updated from Hugging Face. Continuing training with new model...")
                self.model = hf_manager.update_model(hf_manager.repo_id)
                self.optimizer = AdamW(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer

            for step, batch in enumerate(self.data_loader):
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Example of a condition to periodically send gradients
                current_time = time.time()
                if current_time - last_send_time >= 60:  # Example interval
                    self.store_gradients()
                    last_send_time = current_time

class LocalTrainingLoop:
    def __init__(self, model_name, data_loader, learning_rate=5e-5):
        super(LocalTrainingLoop, self).__init__(model_name, data_loader, learning_rate)  

    @staticmethod
    def store_gradients(aggregated_gradients, local_dir="local_gradients", gradient_file_name="gradients.pt"):
        """
        Saves gradients to a file in a specified local directory.
        """
        import os
        # Ensure the local directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Construct the full path to the gradient file
        gradient_file_path = os.path.join(local_dir, gradient_file_name)
        
        # Save gradients to the file
        torch.save(aggregated_gradients, gradient_file_path)
        print(f"Gradients saved locally at: {gradient_file_path}")