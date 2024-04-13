import time
from transformers import AdamW #FIXME replace me with LAMB
from huggingface_hub import Repository
import torch
import math
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer
from bittensor import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import os
import hashlib


class TrainingLoop:
    def __init__(self, model_name, data_loader,gradients_dir, learning_rate=5e-5, send_interval=30, averaging_dir = "averaged_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()

        self.data_loader = data_loader
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.send_interval = send_interval
        self.gradients_dir = gradients_dir
        self.averaging_dir = averaging_dir

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
                self.optimizer = SGD(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer

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
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 10)  # MNIST has 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation, as we'll use CrossEntropyLoss which includes Softmax
        return x

class MNISTTrain(LocalTrainingLoop):
    
    def __init__(self, model_name, data_loader,gradients_dir, test_loader, averaging_dir="averaged_model", learning_rate=5e-5, send_interval=30):
        
        self.model = FeedforwardNN() 
        self.model.train()

        self.data_loader = data_loader
        self.test_loader = test_loader

        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        self.send_interval = send_interval
        self.gradients_dir = gradients_dir
        self.averaging_dir = averaging_dir

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.averaging_dir, exist_ok=True)
        model_save_path = os.path.join(self.averaging_dir, "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

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

    
    def train(self, epochs, hf_manager, n_steps):
        last_send_time = time.time()
        step_counter = 0  # Initialize step counter that persists across epochs
        test_counter = 0
        test_losses = []
        test_accuracies = []
        training_losses = []
        logging.info("Model updated from Hugging Face. Continuing training with new model...")
        #self.model = hf_manager.update_model(self.model)
        self.model = FeedforwardNN()
        
        
        self.optimizer = SGD(self.model.parameters(), lr=0.1)  # Reinitialize the optimizer
        self.optimizer.zero_grad()  # Ensure gradients are reset after model update
        self.aggregated_gradients = {}  # Initialize an empty dictionary for storing aggregated gradients
        for name, param in self.model.named_parameters():  # Iterate over all parameters of the model
            if param.requires_grad:  # Check if the parameter requires gradients and has gradients computed
                self.aggregated_gradients[name] = torch.zeros_like(param)  # Create a zero tensor with the same shape as the parameter        
        
        
        for epoch in range(epochs):
            logging.info(f"Starting Epoch: {epoch}")
            total_loss = 0
            total_examples = 0
            
            

            # if hf_manager.check_for_new_submissions():
            #     logging.info("Model updated from Hugging Face. Continuing training with new model...")
            #     self.model = hf_manager.update_model(self.model)
            #     self.optimizer = SGD(self.model.parameters(), lr=5e-5)  # Reinitialize the optimizer
            #     self.optimizer.zero_grad()  # Ensure gradients are reset after model update

            for batch_idx, (data, target) in enumerate(self.data_loader):
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        self.aggregated_gradients[name] += self.normalize_gradients(param.grad, threshold=0.1)

                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_examples += len(data)

                average_loss = total_loss / total_examples
                #logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")
                

                # Check if it's time to step the optimizer and reset gradients
                if (step_counter + 1) % n_steps == 0:
                    test_counter += 1
                    
                    # for param in self.model.parameters():
                    #     if param.grad is not None:
                    #         param.grad /= (n_steps//10)
                    self.optimizer.zero_grad()

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad = self.aggregated_gradients[name]
                    
                    self.optimizer.step()

                    test_loss, test_accuracy = self.test()
                    #test_losses.append(test_loss)
                    #test_accuracies.append(test_accuracy)
                    train_loss = total_loss / total_examples
                    #training_losses.append(train_loss)
                    logging.info(f"Train Loss: {train_loss} At {step_counter} accumulated gradients")
                    logging.info(f"Test Loss: {test_loss} At {step_counter} accumulated gradients")
                    logging.info(f"Test Accuracy: {test_accuracy} At {step_counter} accumulated gradients")
                    
                    return train_loss, test_loss, test_accuracy


                    self.model.train()
                    
                step_counter += 1  # Increment step counter after processing each batch

                # Periodic actions such as logging and sending gradients
                if time.time() - last_send_time >= self.send_interval:
                    average_loss = total_loss / total_examples
                    logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")

                    # Logic to send aggregated gradients
                    self.aggregated_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.requires_grad and param.grad is not None}
                    self.store_gradients(self.aggregated_gradients, self.gradients_dir)

                    last_send_time = time.time()
                    #total_loss = 0
                    #total_examples = 0  # Reset for the next interval

    
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


class MNISTDeltaTrain(LocalTrainingLoop):
    
    def __init__(self, model_name, data_loader,gradients_dir, test_loader, averaging_dir="averaged_model", learning_rate=5e-5, send_interval=30):
        
        self.model = FeedforwardNN() 
        self.model.train()

        self.data_loader = data_loader
        self.test_loader = test_loader

        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        self.send_interval = send_interval
        self.gradients_dir = gradients_dir
        self.averaging_dir = averaging_dir

    def save_model(self):
        """
        Saves the model to the specified local directory.
        """
        os.makedirs(self.averaging_dir, exist_ok=True)
        model_save_path = os.path.join(self.averaging_dir, "averaged_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model saved locally at {model_save_path}.")

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

    
    def train(self, epochs, hf_manager, n_steps):
        last_send_time = time.time()
        step_counter = 0  # Initialize step counter that persists across epochs
        test_counter = 0
        test_losses = []
        test_accuracies = []
        training_losses = []
        logging.info("Model updated from Hugging Face. Continuing training with new model...")
        #self.model = hf_manager.update_model(self.model)
        self.model = FeedforwardNN()
        
        
        self.optimizer = SGD(self.model.parameters(), lr=0.1)  # Reinitialize the optimizer
        self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()}        
        
        for epoch in range(epochs):
            logging.info(f"Starting Epoch: {epoch}")
            total_loss = 0
            total_examples = 0

            if hf_manager.check_for_new_submissions():
                logging.info("Model updated from Hugging Face. Continuing training with new model...")
                self.model = hf_manager.update_model(self.model)
                self.optimizer = SGD(self.model.parameters(), lr=0.1)  # Reinitialize the optimizer
                self.base_weights = {name: param.clone() for name, param in self.model.named_parameters()}
                #self.optimizer.zero_grad()  # Ensure gradients are reset after model update

            for batch_idx, (data, target) in enumerate(self.data_loader):
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_examples += len(data)

                average_loss = total_loss / total_examples
                #logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")
                

                # Check if it's time to step the optimizer and reset gradients
                if (step_counter + 1) % n_steps == 0:
                    test_counter += 1                    
                    
                    test_loss, test_accuracy = self.test()
                    #test_losses.append(test_loss)
                    #test_accuracies.append(test_accuracy)
                    train_loss = total_loss / total_examples
                    #training_losses.append(train_loss)
                    logging.info(f"Train Loss: {train_loss} At {step_counter} accumulated gradients")
                    logging.info(f"Test Loss: {test_loss} At {step_counter} accumulated gradients")
                    logging.info(f"Test Accuracy: {test_accuracy} At {step_counter} accumulated gradients")
                    
                    #return train_loss, test_loss, test_accuracy

                    self.model.train()
                    
                step_counter += 1  # Increment step counter after processing each batch

                # Periodic actions such as logging and sending gradients
                if time.time() - last_send_time >= self.send_interval:
                    average_loss = total_loss / total_examples
                    logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {average_loss:.4f}")

                    # Logic to send aggregated gradients
                    self.weight_diffs = {name:  param.data - self.base_weights[name] for name, param in self.model.named_parameters() if param.requires_grad}
                    self.store_gradients(self.weight_diffs, self.gradients_dir)


                    logging.info(f"Model hash is: {self.calculate_model_hash()}")

                    last_send_time = time.time()
                    #total_loss = 0
                    #total_examples = 0  # Reset for the next interval

    
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