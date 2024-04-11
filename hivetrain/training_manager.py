import time
from transformers import AdamW #FIXME replace me with LAMB
from huggingface_hub import Repository
import torch
import math
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from bittensor import logging
import torch.nn as nn
import torch.nn.functional as F

class TrainingLoop:
    def __init__(self, model_name, data_loader,gradients_dir, learning_rate=5e-5, send_interval=30):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()

        self.data_loader = data_loader
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.send_interval = send_interval
        self.gradients_dir = gradients_dir

    @staticmethod
    def store_gradients(aggregated_gradients, hf_repo_path, gradient_file_name="gradients.pt", use_auth_token=True):
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
        self.aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        for epoch in range(epochs):
            logging.info(f"Starting Epoch: {epoch}")
            # Check for new submissions at the start of each epoch

            total_loss = 0
            total_examples = 0

            if hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = hf_manager.update_model(self.model)
                self.optimizer = AdamW(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer

            for step, batch in enumerate(self.data_loader):
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()

                # Update loss and example counts
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_examples += batch['input_ids'].size(0)

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.aggregated_gradients[name] += param.grad
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Example of a condition to periodically send gradients
                current_time = time.time() ##time.time()%sth works better
                if current_time - last_send_time >= self.send_interval:
                    
                    average_loss = total_loss / total_examples
                    perplexity = math.exp(average_loss)
                    logging.info(f"Epoch: {epoch}, Examples: {total_examples}, Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}")

                    try:
                        logging.info(f"Attempting to send gradients")

                        self.store_gradients(self.aggregated_gradients, self.gradients_dir)
                    except Exception as e:
                        logging.warning(f"Sending gradients failed: {e}")
                        continue
                    last_send_time = current_time
                    # Reset aggregated gradients
                    #self.aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

class LocalTrainingLoop(TrainingLoop):
    def __init__(self, model_name, data_loader, learning_rate=5e-5):
        super(LocalTrainingLoop, self).__init__(model_name, data_loader, learning_rate)  

    @staticmethod
    def store_gradients(aggregated_gradients, local_dir, gradient_file_name="gradients.pt"):
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


class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)  # Flatten 28x28 images to a 784 vector
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)  # MNIST has 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, as we'll use CrossEntropyLoss which includes Softmax
        return x

class MNISTTrain(LocalTrainingLoop):
    
    def __init__(self, model_name, data_loader,gradients_dir, learning_rate=5e-5, send_interval=30):
        
        self.model = FeedforwardNN() 
        self.model.train()

        self.data_loader = data_loader
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.send_interval = send_interval
        self.gradients_dir = gradients_dir
    
    def train(self, epochs, hf_manager):
        last_send_time = time.time()
        self.aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        for epoch in range(epochs):
            logging.info(f"Starting Epoch: {epoch}")
            total_loss = 0
            total_examples = 0

            if hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = hf_manager.update_model(self.model)
                self.optimizer = AdamW(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer
                self.aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

            for batch_idx, (data, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.aggregated_gradients[name] += param.grad

                total_loss += loss.item()
                total_examples += len(data)

                if time.time() - last_send_time >= self.send_interval:
                    average_loss = total_loss / total_examples
                    logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")
                    # Example logic to send gradients or perform any other periodic action
                    # Reset for next 
                    self.store_gradients(self.aggregated_gradients, self.gradients_dir)
                    last_send_time = time.time()
                    total_loss = 0
                    total_examples = 0