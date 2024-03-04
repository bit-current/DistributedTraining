import subprocess
import logging
import os
import signal

class SubprocessHandler:
    def __init__(self):
        self.process = None

    def start_tcp_store(self, store_address, store_port, timeout=30):
        try:
            command = ['python', 'tcp_store_server.py', store_address, str(store_port), str(timeout)]
            self.process = subprocess.Popen(command, shell=False)
            logging.info(f"TCPStore server started at {store_address}:{store_port} with timeout {timeout}")
        except Exception as e:
            logging.error(f"Failed to launch TCPStore subprocess: {e}")
            self.process = None

    def check_process(self):
        if self.process and self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            if self.process.returncode == 0:
                logging.info(f"TCPStore subprocess exited successfully: {stdout.decode()}")
            else:
                logging.error(f"TCPStore subprocess exited with errors: {stderr.decode()}")

    def cleanup(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(self.process.pid, signal.SIGKILL)
            logging.info("TCPStore subprocess cleaned up successfully")

