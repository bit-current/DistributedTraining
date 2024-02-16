# validator.py
from flask import Flask, request

app = Flask(__name__)

@app.route('/validate_gradient', methods=['POST'])
def validate_gradient():
    data = request.get_json()
    print(f"Received gradient checksum: {data['checksum']}")
    return "OK"

@app.route('/log_metrics', methods=['POST'])
def log_metrics():
    metrics = request.get_json()
    print(f"Received metrics: {metrics}")
    return "OK"

@app.route('/verify_model', methods=['POST'])
def verify_model():
    data = request.get_json()
    print(f"Received model checksum from rank {data['rank']}: {data['checksum']}")
    return "OK"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
