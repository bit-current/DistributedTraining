import io
import math 
import threading

class ModelValidator:
    def __init__(self, model, data_loader, criterion, bittensor_network = None, interval=3600):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.interval = interval  # Validation interval in seconds
        self.original_state_dict = model.state_dict()
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.scores = {}
        self.bittensor_network = bittensor_network

    @staticmethod
    def deserialize_gradients(serialized_gradients):
        buffer = io.BytesIO(serialized_gradients)
        buffer.seek(0)
        return torch.load(buffer)

    @staticmethod
    def receive_gradients(storage_dht = dht_manager.my_dht, hotkey = my_hotkey):
        serialized_gradients = storage_dht.get(hotkey)
        aggregated_gradients = deserialize_gradients(serialized_gradients)
        
        return aggregated_gradients

    def receive_gradients(self):
        # Get validators uids
        if time.time() - self.last_sync_time > self.sync_interval:
            sync(self.last_sync_time, self.sync_interval, BittensorNetwork.config)#scope issue FIXME?
            self.last_sync_time = time.time()

        validator_uids = self.bittensor_network.get_validator_uids()
        # Get average of validator weights weighted by their stake?
        self.miner_gradients = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            if uid not in validator_uids:
                try:
                    gradient = receive_gradients(self.dht, hotkey)
                    miner_gradients.append(miner_gradients)
                except:
                    self.miner_gradients.append(None)

    def update_model_weights(self, gradients):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= gradients[name]

    def evaluate_model(self, metric='loss'):
        #WARNING: # 2. The evaluate_model method incorrectly uses the criterion on outputs directly. 
        # If the model's outputs are logits, and the criterion expects logits and labels, this might be correct, 
        # but typically, a transformation is applied to outputs before calculating loss (e.g., softmax for cross-entropy loss).
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in self.data_loader:
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                labels = batch['input_ids']
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        perplexity = math.exp(average_loss) if metric == 'perplexity' else None
        return average_loss, perplexity

    def validate_and_score(self):
        
        for hotkey_address in self.bittensor_network.metagraph.hotkeys:
            gradients = receive_gradients(hotkey_address)
            
            self.update_model_weights(gradients)
            loss, perplexity = self.evaluate_model()
            loss_score = max(0, self.base_loss - loss)
            perplexity_score = max(0, self.base_perplexity - perplexity) if perplexity else 0
            self.scores[hotkey_address] = perplexity_score

            # Reset the model to its original state
            self.model.load_state_dict(self.original_state_dict)
            
            print(f"Loss Score: {loss_score}, Perplexity Score: {perplexity_score}")

    def start_periodic_validation(self):
        def run():
            while True:
                self.validate_and_score()
                time.sleep(self.interval)
        
        threading.Thread(target=run, daemon=True).start()
