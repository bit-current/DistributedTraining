import argparse
import subprocess
from time import sleep

def start_meta_miner(miner_script, orchestrator_url, batch_size, epochs, port, hotkey, tcp_store_port, netuid=1, chain_endpoint="ws://127.0.0.1:9946", wallet_name="torch_local_cold", host_address="127.0.0.1", vpermit_tao_limit=10, tcp_store_address="127.0.0.1"):
    """Start a miner process with the given configuration."""
    cmd = [
        "python", miner_script,
        "--netuid", str(netuid),
        "--orchestrator-url", orchestrator_url,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--subtensor.chain_endpoint", chain_endpoint,
        "--wallet.name", wallet_name,
        "--wallet.hotkey", hotkey,
        "--host-address", host_address,
        "--port", str(port),
        "--neuron.vpermit_tao_limit", str(vpermit_tao_limit),
        "--tcp-store-address", tcp_store_address,
        "--tcp-store-port", str(tcp_store_port),
    ]
    return subprocess.Popen(cmd, shell=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch meta miners with configurable port and hotkey ranges.")
    parser.add_argument("--start-port", type=int, required=True, help="Starting port number for the miners.")
    parser.add_argument("--end-port", type=int, required=True, help="Ending port number for the miners.")
    parser.add_argument("--start-hotkey-number", type=int, required=True, help="Starting number for hotkey suffix.")
    parser.add_argument("--end-hotkey-number", type=int, required=True, help="Ending number for hotkey suffix.")
    parser.add_argument("--start-tcp-store-port", type=int, required=True, help="Starting TCP store port number.")
    parser.add_argument("--end-tcp-store-port", type=int, required=True, help="Ending TCP store port number.")
    args = parser.parse_args()

    miner_script = "miner_cpu_simple.py"
    batch_size = 1
    epochs = 1
    orchestrator_url = "http://127.0.0.1:5000"
    netuid = 1
    chain_endpoint = "ws://127.0.0.1:9946"
    wallet_name = "torch_local_cold"
    host_address = "127.0.0.1"
    vpermit_tao_limit = 10

    processes = []
    for port, hotkey_number, tcp_store_port in zip(range(args.start_port, args.end_port + 1), range(args.start_hotkey_number, args.end_hotkey_number + 1), range(args.start_tcp_store_port, args.end_tcp_store_port + 1)):
        hotkey = f"torch_local_hot{hotkey_number}"
        process = start_meta_miner(miner_script, orchestrator_url, batch_size, epochs, port, hotkey, tcp_store_port, netuid, chain_endpoint, wallet_name, host_address, vpermit_tao_limit)
        processes.append(process)
        sleep(2)
        print(f"Launched meta_miner with {hotkey} on port {port} and TCP store port {tcp_store_port}")

    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        # Terminate all processes if script is interrupted
        for process in processes:
            process.terminate()
        print("Terminated all miner processes.")
