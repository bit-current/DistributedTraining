import subprocess
from time import sleep

def start_miner(rank, world_size=3, batch_size=512, epochs=100, validator_urls=[]):
    """Start a miner process with the given configuration."""
    cmd = [
        "python", "miner_cpu_simple.py",
        "--world-size", str(world_size),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
    ] + ["--validator-urls"] + validator_urls + ["--rank", str(rank)]
    return subprocess.Popen(cmd, shell=False)

if __name__ == "__main__":
    world_size = 3  # Total number of miners
    batch_size = 512
    epochs = 100
    ports = [i for i in range(5001, 5010)]  # List of ports for validators
    validator_urls = [f"http://127.0.0.1:{port}" for port in ports]

    processes = []
    for rank in range(world_size):
        processes.append(start_miner(rank, world_size, batch_size, epochs, validator_urls))
        sleep(10)
        print(f"Launched miner {rank}")

    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        # Terminate all processes if script is interrupted
        for process in processes:
            process.terminate()
        print("Terminated all miner processes.")
