import subprocess

def start_validator(port, host_address):
    """Start a validator process on the given port and host address."""
    cmd = [
        "python", "validator.py", 
        "--netuid", "1", 
        "--orchestrator-url", "http://127.0.0.1:5000", 
        "--batch-size", "1", 
        "--epochs", "1", 
        "--miner-script", "miner_cpu_simple.py", 
        "--subtensor.chain_endpoint", "ws://127.0.0.1:9946", 
        "--wallet.name", "torch_local_cold2", 
        "--wallet.hotkey", "torch_local_hot22", 
        "--host-address", host_address, 
        "--port", str(port), 
        "--neuron.vpermit_tao_limit", "10", 
        "--axon.port", str(port), 
        "--axon.ip", host_address, 
        "--axon.external_port", str(port), 
        "--axon.external_ip", host_address
    ]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    host_address = "127.0.0.1"  # Set the host address
    ports = [i for i in range(3000, 3009 + 1)]  # Range of ports for validators
    processes = [start_validator(port, host_address) for port in ports]

    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        # Terminate all processes if script is interrupted
        for process in processes:
            process.terminate()
        print("Terminated all validator processes.")
