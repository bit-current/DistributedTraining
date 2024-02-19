import argparse
import requests
import time
import subprocess
import json
import atexit
import signal


def register_with_orchestrator(orchestrator_url):
    """Attempt to register the miner with the orchestrator."""
    response = requests.post(f"{orchestrator_url}/register")
    if response.status_code == 200:
        data = response.json()
        return data['miner_id'], data['state']
    else:
        return None, response.json().get('error', 'Failed to register')

def update_orchestrator(orchestrator_url, miner_id, trigger_error=False):
    """Send an update to the orchestrator, optionally with an error trigger."""
    data = {'miner_id': miner_id, 'trigger_error': trigger_error}
    response = requests.post(f"{orchestrator_url}/update", json=data)
    return response.json() if response.ok else None

def get_training_params(orchestrator_url, miner_id):
    """Retrieve training parameters from the orchestrator."""
    params = {'miner_id': miner_id}
    response = requests.get(f"{orchestrator_url}/training_params", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def start_training(rank, world_size, miner_script, batch_size, epochs, validator_urls, store_address, store_port):
    # Ensure all command line arguments are strings
    cmd = [
        "python", miner_script,
        f"--rank={str(rank)}",
        f"--epochs={str(epochs)}",
        f"--world-size={str(world_size)}",
        f"--batch-size={str(batch_size)}",
        f"--store-address={store_address}",
        f"--store-port={str(store_port)}",
    ] + ["--validator-urls"] + validator_urls
    return subprocess.Popen(cmd, shell=False)

def cleanup_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    gone, still_alive = psutil.wait_procs(children, timeout=3, callback=None)
    for alive in still_alive:
        alive.kill()

def main(orchestrator_url, miner_script, batch_size, epochs, validator_urls):
    miner_id = None
    torchrun_process = None

    while True:
        # Always get the current state from the orchestrator
        update_response = update_orchestrator(orchestrator_url, miner_id)
        if update_response:
            state = update_response.get("state")
            miner_id = update_response.get("miner_id", miner_id)  # Update miner_id if provided
            print(f"Current state from orchestrator: {state}")

            if state == "training" and not torchrun_process:
                training_params = get_training_params(orchestrator_url, miner_id)
                if training_params:
                    print(f"Starting training with params: {training_params}")
                    # Extract and use training parameters
                    torchrun_process = start_training(training_params['rank'], training_params['world_size'], miner_script, batch_size, epochs, validator_urls, "127.0.0.1", "4999")
            elif state == "onboarding" and torchrun_process:
                # If the state has reverted to onboarding, terminate any running training process
                torchrun_process.terminate()
                torchrun_process.wait()
                torchrun_process = None
                print("Training process terminated, ready for next run.")
        else:
            # Attempt to register if not already done
            if miner_id is None:
                miner_id, state = register_with_orchestrator(orchestrator_url)
                if miner_id:
                    print(f"Registered with miner ID: {miner_id}, State: {state}")
                else:
                    print("Failed to register with orchestrator.")

        time.sleep(10)  # Polling interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta Miner Configuration")
    parser.add_argument("--orchestrator-url", type=str, required=True, help="URL of the orchestrator")
    parser.add_argument("--miner-script", type=str, default="miner_cpu.py", help="The miner script to execute for training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per forward/backward pass")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--validator-urls', type=str, nargs="+", required=True, help='URLs of the validators')

    args = parser.parse_args()
    main(args.orchestrator_url, args.miner_script, args.batch_size, args.epochs, args.validator_urls)