import threading
import requests
import json
import hashlib

def send_request_in_background(url, data):
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"Failed to send data to {url}: {e}")

def send_model_checksum(model, rank, validator_urls):
    model_state = model.state_dict()
    model_bytes = json.dumps({k: v.tolist() for k, v in model_state.items()}, sort_keys=True).encode()
    checksum = hashlib.md5(model_bytes).hexdigest()
    data = {"rank": rank, "checksum": checksum}
    
    for url in validator_urls:
        thread = threading.Thread(target=send_request_in_background, args=(url, data))
        thread.start()

def send_metrics(metrics, rank, validator_urls):
    metrics_data = json.dumps(metrics, sort_keys=True, ensure_ascii=True)
    checksum = hashlib.md5(metrics_data.encode()).hexdigest()
    data = {"rank": rank, "checksum": checksum, "metrics": metrics}
    
    for url in validator_urls:
        thread = threading.Thread(target=send_request_in_background, args=(url, data))
        thread.start()

# Modify the hook method of the ValidateGradientHook class to use threading as well
class ValidateGradientHook:
    def __init__(self, validator_urls):
        self.validator_urls = validator_urls

    def hook(self, state, bucket):
        for tensor in bucket.get_tensors():
            checksum = hashlib.md5(tensor.numpy()).hexdigest()
            data = {"checksum": checksum}
            for url in self.validator_urls:
                thread = threading.Thread(target=send_request_in_background, args=(url, data))
                thread.start()
        return bucket.get_tensors()