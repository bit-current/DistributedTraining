import subprocess
from time import sleep

def start_meta_miner(miner_script, orchestrator_url, batch_size=512, epochs=100, validator_urls=[]):
    """Start a miner process with the given configuration."""
    cmd = [
        "python", "meta_miner.py",
        "--miner-script", miner_script,
        "--orchestrator-url", orchestrator_url,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
    ] + ["--validator-urls"] + validator_urls 
    return subprocess.Popen(cmd, shell=False)



if __name__ == "__main__":
    miner_script = "miner_cpu_simple.py"
    batch_size = 512
    epochs = 100
    orchestrator_url = "http://127.0.0.1:5000"
    ports = [i for i in range(5001, 5010)]  # List of ports for validators
    validator_urls = [f"http://127.0.0.1:{port}" for port in ports]
    meta_miners_to_launch = 3 

    processes = []
    for count in range(meta_miners_to_launch):
        processes.append(start_meta_miner(miner_script, orchestrator_url, batch_size, epochs, validator_urls))
        sleep(2)
        print(f"Launched meta_miner {count}")

    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        # Terminate all processes if script is interrupted
        for process in processes:
            process.terminate()
        print("Terminated all miner processes.")

