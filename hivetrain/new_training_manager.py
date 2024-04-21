import time
from transformers import AdamW #FIXME replace me with LAMB
from huggingface_hub import Repository
import torch
import math
from dotenv import load_dotenv
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from bittensor import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.optim as optim
import os
import hashlib

load_dotenv()
token = os.getenv("HF_TOKEN")


class TrainingLoopNew:
    def __init__(self, model, device, hf_manager, train_loader,test_loader, send_interval = 120, check_update_interval = 60, learning_rate=5e-5,):
        self.model = model.to(device)
        self.device = device
        self.hf_manager = hf_manager
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.send_interval = send_interval
        self.check_update_interval = check_update_interval
        self.learning_rate = learning_rate
    
    def train(self, epochs, n_steps):
        total_loss = 0
        total_examples = 0
        step_counter = 0  # Initialize step counter that persists across epochs
        test_counter = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()} 

        self.last_pull_time = time.time()
        self.last_send_time = time.time()

        for epoch in range(epochs):
            
            print("************** NEW EPOCH")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                if time.time() - self.last_pull_time >= self.check_update_interval and self.hf_manager.check_for_new_submissions(self.hf_manager.model_repo_id):
                    logging.info("Averaged model updated on Hugging Face. Pulling latest model...")
                    print("********Averaged model updated on Hugging Face. Pulling latest model...")
                    self.hf_manager.pull_latest_model()
                    time.sleep(10) #just to give enough time for pull
                    self.model = self.hf_manager.update_model(self.model)
                    optimizer = optim.Adam(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer
                    self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()} 
                    self.last_pull_time = time.time()

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_examples += len(data)

                average_loss = total_loss / total_examples
                #logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")
                

                # Check if it's time to step the optimizer and reset gradients
                if (step_counter + 1) % n_steps == 0:
                    test_counter += 1                    
                    
                    test_loss, test_accuracy = self.test()
                    train_loss = total_loss / total_examples
                    logging.info(f"Train Loss: {train_loss} At {step_counter} accumulated gradients")
                    print("***Train Loss: {train_loss} At {step_counter} accumulated gradients")

                    logging.info(f"Test Loss: {test_loss} At {step_counter} accumulated gradients")
                    logging.info(f"Test Accuracy: {test_accuracy} At {step_counter} accumulated gradients")
                    print((f"Test Accuracy: {test_accuracy} At {step_counter} accumulated gradients"))
                    
                    #return train_loss, test_loss, test_accuracy
                    self.model.train()
                    
                step_counter += 1  # Increment step counter after processing each batch
                # Periodic actions such as logging and sending gradients
                if time.time() - self.last_send_time >= self.send_interval:
                    average_loss = total_loss / total_examples
                    perplexity = math.exp(average_loss)
                    logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")

                    try:
                        logging.info(f"Attempting to send weights")
                        logging.info(f"********* Attempting to send weights")
                        # Periodically save gradients
                        model_gradients_path = os.path.join(self.hf_manager.get_local_gradient_directory(), 'weight_diff.pt')
                        self.weight_diffs = {name:  param.data - self.base_weights[name] for name, param in self.model.named_parameters() if param.requires_grad}
                        torch.save(self.weight_diffs, model_gradients_path)
                        self.hf_manager.push_changes(['weight_diff.pt'])
                    except Exception as e:
                        logging.warning(f"Sending gradients failed: {e}")
                        continue

                    logging.info(f"Model hash is: {self.calculate_model_hash()}")
                    print(f"Model hash is: {self.calculate_model_hash()}")
                    self.last_send_time = time.time()
                
                if batch_idx % 50 == 0:  # For example, save every 50 batches
                    print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

   
    def calculate_model_hash(self): 
        model_hash = hashlib.sha256()
        for name, param in self.model.named_parameters():
            model_hash.update(name.encode('utf-8'))
            model_hash.update(param.data.cpu().numpy().tobytes())
        return model_hash.hexdigest()
   
    def test(self):
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        total_test_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:

                images, labels = batch
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item() 
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_test_samples += labels.size(0)

        average_test_loss = test_loss / total_test_samples
        accuracy = correct_predictions / total_test_samples
        return average_test_loss, accuracy


    @staticmethod
    def normalize_gradients(parameter, threshold=1.0):
        """
        Normalize the gradients to avoid exploding or vanishing gradients.
        
        Args:
        parameters (iterable): Iterable of model parameters (typically model.parameters() in PyTorch).
        threshold (float): The maximum norm value for gradients. Defaults to 1.0.
        """
        param_norm = parameter.norm(2)
        
        # Normalize if the total norm exceeds the threshold
        if param_norm > threshold:
            return parameter.data.mul_(threshold / param_norm)
        else:
            return parameter

    def calculate_model_hash(self): 
        model_hash = hashlib.sha256()
        for name, param in self.model.named_parameters():
            model_hash.update(name.encode('utf-8'))
            model_hash.update(param.data.cpu().numpy().tobytes())
        return model_hash.hexdigest()


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)
