# tcp_store_server.py
import sys
import logging
from torch.distributed import TCPStore
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(store_address, store_port, timeout=30):
    # Convert port to integer
    
    store_port = int(store_port)
    timeout = int(timeout)
    try:
        # Initialize the TCPStore
        breakpoint()
        store = TCPStore(store_address, store_port, None, True,timedelta(seconds=timeout))
        logging.info(f"TCPStore running at {store_address}:{store_port}")

        # Keep the store running indefinitely
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logging.info("Shutting down TCPStore.")
    except Exception as e:
        logging.error(f"Failed to start TCPStore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python tcp_store_server.py <store_address> <store_port> <timeout_interval>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
