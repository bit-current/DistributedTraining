class ModelValidator:
    def __init__(self, model, data_loader, criterion, interval=3600):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.interval = interval  # Validation interval in seconds
        self.original_state_dict = model.state_dict()
        self.base_loss, self.base_perplexity = self.evaluate_model()
        self.scores = {}

    def deserialize_gradients(self, serialized_gradients):
        buffer = io.BytesIO(serialized_gradients)
        buffer.seek(0)
        return torch.load(buffer)

    def update_model_weights(self, gradients):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= gradients[name]

    def evaluate_model(self, metric='loss'):
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
        
        for hotkey_address in BittensorNetwork.metagraph.hotkeys:
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
