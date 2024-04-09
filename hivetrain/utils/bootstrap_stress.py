import argparse
import asyncio
import aiohttp
import random
import time

async def ping_server(session, server_url):
    start_time = time.time()
    async with session.get(server_url) as response:
        if response.status == 200:
            dht_address = await response.text()
            end_time = time.time()
            latency = end_time - start_time
            print(f"Received DHT address: {dht_address}, Latency: {latency:.4f} seconds")
        else:
            print(f"Request failed with status code: {response.status}")

async def stress_test(server_url, num_requests, concurrent_requests):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            task = asyncio.create_task(ping_server(session, server_url))
            tasks.append(task)
            if len(tasks) >= concurrent_requests:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

async def run_stress_test(server_url, num_requests, concurrent_requests, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        await stress_test(server_url, num_requests, concurrent_requests)
        print(f"Completed iteration at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    end_time = time.time()
    print(f"\nStress test completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stress Test Client')
    parser.add_argument('--server_url', type=str, default="http://localhost:5000/return_dht_address",
                        help='Server URL to ping')
    parser.add_argument('--num_requests', type=int, default=100, help='Total number of requests per iteration')
    parser.add_argument('--concurrent_requests', type=int, default=50,
                        help='Number of concurrent requests to send')
    parser.add_argument('--duration', type=int, default=300, help='Duration of the stress test in seconds (default: 1800 = 30 minutes)')
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.server_url, args.num_requests, args.concurrent_requests, args.duration))