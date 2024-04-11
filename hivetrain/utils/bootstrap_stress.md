 # Stress Test Client Documentation

This Python script is designed to perform a stress test on a given server by sending multiple concurrent HTTP requests and measuring the response time. The following components make up the logic of this script:

## Import Statements
```python
import argparse
import asyncio
import aiohttp
import random
import time
```
The import statements load several Python modules required to run the stress test. `argparse` is used for parsing command-line arguments, `asyncio` and `aiohttp` handle asynchronous tasks andMAGE_TARGET_URL requests, respectively, while `random` and `time` are used for randomizing request order and measuring time elapsed.

## `ping_server` Function (Async)
```python
async def ping_server(session, server_url):
    ...
```
The `ping_server` function sends a single HTTP GET request to the provided server URL using an asynchronous session created with `aiohttp.ClientSession()`. If the response status code is 200 OK, it extracts the DHT address from the server's response and calculates the latency time. The function prints the received DHT address and its corresponding latency to the console.

## `stress_test` Function (Async)
```python
async def stress_test(server_url, num_requests, concurrent_requests):
    ...
```
The `stress_test` function is responsible for sending multiple requests in parallel. It creates a given number of tasks each running the `ping_server` function and limits the active number of tasks based on the specified `concurrent_requests`. Once the limit is reached, it waits for all finished tasks to complete before starting new ones. This process repeats until all the required number of requests (`num_requests`) have been sent.

## `run_stress_test` Function (Async)
```python
async def run_stress_test(server_url, num_requests, concurrent_requests, duration):
    ...
```
The `run_stress_test` function is the main entry point of the script. It sets up the server URL, number of requests, and concurrent requests based on the command-line arguments provided. The function then runs an asynchronous loop that continues for the specified test duration. During each iteration of the loop, it calls `stress_test` with the given configuration to send the required number of requests in parallel. The loop prints a message at the end of each completed iteration, and finally, it prints a summary message once the stress test has ended.

## Command-Line Arguments (argparse)
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stress Test Client')
    ...
```
The script's main logic is wrapped in an `if __name__ == '__main__'` block, which initializes the `argparse` parser and sets up various arguments such as `--server_url`, `--num_requests`, `--concurrent_requests`, and `--duration`. These arguments can be provided when running the script from the command line to customize its behavior. Finally, it calls `asyncio.run()` to execute the `run_stress_test` function asynchronously using the provided arguments.