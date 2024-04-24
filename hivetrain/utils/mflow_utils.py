import os
import torch
import psutil
import time
import re


def get_gpu_utilization():
    """
    Retrieves the GPU utilization as a percentage of the GPU's capacity.

    Returns:
        float: The GPU utilization percentage if CUDA is available, otherwise 0.0.

    """
    # Check if the CUDA is available on the current device using PyTorch's built-in function
    if torch.cuda.is_available():
        # Retrieve and return the GPU utilization percentage
        utilization = torch.cuda.utilization()
        return utilization
    else:
        # Return 0.0 if CUDA is not available, indicating no GPU activity
        return 0.0


def get_memory_usage():
    """
    Retrieves the current memory usage of the process that runs a Python script.

    Returns:
        int: The current memory usage (RSS) of the process in megabytes.

    """
    # Create a process instance for the current process using psutil
    process = psutil.Process()
    # Retrieve memory info from the current process
    memory_info = process.memory_info()
    # Return the Resident Set Size (RSS) which indicates how much memory in megabytes the process is using in RAM
    return round(memory_info.rss / 1024**2, 2)


def get_network_bandwidth():
    """
    Retrieves the total network bandwidth usage by calculating the sum of bytes sent and received.

    Returns:
        int: The total number of bytes sent and received over the network.

    """
    net_io_counters = psutil.net_io_counters()
    return net_io_counters.bytes_sent + net_io_counters.bytes_recv


def get_version_from_file():
    file_path = os.path.join(os.getcwd(), "template/__init__.py")
    # Read the specified file and search for version information
    with open(file_path, "r") as file:
        content = file.read()
    # Regex to match __version__ = 'x.y.z'
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    if version_match:
        return version_match.group(1)
    else:
        return None


VERSION = get_version_from_file()
