# By Kobetryz

import torch
import psutil
import time


def get_gpu_utilization():
    """
    Retrieves the GPU utilization as a percentage of the GPU's capacity.

    This function checks if CUDA (NVIDIA's GPU computing API) is available on the system using PyTorch's capabilities.
    If CUDA is available, it fetches the current GPU utilization percentage which indicates how much of the GPU's
    computing power is currently being used. If CUDA is not available, indicating that either there are no CUDA-capable
    devices on the system or PyTorch is not configured to use them, the function returns 0.0, suggesting no GPU utilization.

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
    Retrieves the current memory usage of the process that runs this Python script.

    This function utilizes the `psutil` library to access system and process utilities. It specifically fetches the
    memory usage statistics of the currently running process, which is the process executing this script. The key metric
    returned is the Resident Set Size (RSS), which is the portion of memory occupied by a process that is held in RAM.

    Returns:
        int: The current memory usage (RSS) of the process in megabytes.

    """
    # Create a process instance for the current process using psutil
    process = psutil.Process()
    # Retrieve memory info from the current process
    memory_info = process.memory_info()
    # Return the Resident Set Size (RSS) which indicates how much memory in megabytes the process is using in RAM
    return round(memory_info.rss/1024**2, 2)
