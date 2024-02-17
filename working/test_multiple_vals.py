import subprocess

def start_validator(port):
    """Start a validator process on the given port."""
    cmd = ["python", "validator.py", "--port", str(port)]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    ports = [i for i in range(5001,5010)]  # List of ports for validators
    processes = [start_validator(port) for port in ports]

    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        # Terminate all processes if script is interrupted
        for process in processes:
            process.terminate()
        print("Terminated all validator processes.")