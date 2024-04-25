import os
import torch
import psutil
import time
import re
import logging
import mlflow
from hivetrain.config.mlflow_config import MLFLOW_UI_URL
import requests
from requests.adapters import HTTPAdapter
from requests.sessions import Session
from urllib3.util.retry import Retry


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


def get_cpu_utilization():
    """
    Returns the current system-wide CPU utilization as a percentage.

    Returns:
        float: The percentage of CPU utilization.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage


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


def initialize_mlflow(
    role,
    device,
    version,
    mlflow_ui_url,
    current_model_name,
    my_hotkey = None,
    learning_rate=None,
    send_interval=None,
    check_update_interval=None,
):
    try:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        os.environ["MLFLOW_START_RETRY_ATTEMPT_MAX"] = "2"
        mlflow.set_tracking_uri(mlflow_ui_url)
        mlflow.set_experiment(current_model_name)

        if role == "miner":
            run_name = f"miner_{my_hotkey}"
            mlflow.start_run(run_name=run_name)
            mlflow.log_param("device", device)
            mlflow.log_param("Version of Code", version)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("send_interval", send_interval)
            mlflow.log_param("check_update_interval", check_update_interval)
        elif role == "validator":
            run_name = f"validator_{my_hotkey}"
            mlflow.start_run(run_name=run_name)
            mlflow.log_param("device", device)
            mlflow.log_param("Version of Code", version)
            mlflow.log_param("check_update_interval", check_update_interval)
        else:
            run_name = f"AVERAGER"
            mlflow.start_run(run_name=run_name)
            mlflow.log_param("device", device)
            mlflow.log_param("Version of Code", version)
    except Exception as e:
        logging.error(f"Failed to initialize and log parameters to MLflow: {e}")
        return None


def log_model_metrics(step, **metrics):
    """
    Logs given metrics to MLflow with the provided step count/int(time).

    Args:
        step (int): The step count or timestamp at which metrics are logged, providing a timeline for metrics.
        **metrics (dict): Arbitrary number of keyword arguments where keys are the metric names and values are their respective values to log.
    """
    try:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)
        print(f"Logged metrics to MLflow at Step {step}: {metrics}")
    except Exception as e:
        logging.error(f"Failed to log metrics to MLflow: {e}")
        print(f"Error logging to MLflow, but training continues: {e}")


def setup_mlflow_session(
    mlflow_tracking_uri, retries=2, backoff_factor=1, status_forcelist=(500, 502, 504)
):
    """
    Sets up a custom requests session for MLflow with specified retry logic.

    Args:
    mlflow_tracking_uri (str): The MLflow server's tracking URI.
    retries (int): The number of retries for requests.
    backoff_factor (float): A backoff factor to apply between attempts.
    status_forcelist (tuple): A set of HTTP status codes that we should force a retry on.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.utils.rest_utils.http_request(session=session)


def create_mlflow_session():
    """Creates a requests session for MLflow with custom retry behavior."""
    session = Session()
    retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


VERSION = get_version_from_file()
