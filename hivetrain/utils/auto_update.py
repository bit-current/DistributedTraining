import os
import subprocess
import time
import requests

def run_script(repo_dir):
    original_dir = os.getcwd()  # Save the original working directory
    
    repo_url = "https://github.com/username/repo.git"
    
    if os.path.exists(repo_dir):
        # Remove the existing directory
        subprocess.run(["rm", "-rf", repo_dir])
    
    
    # Change to the repository directory (if not already there)
    os.chdir(repo_dir)
    
    # Install the package using pip
    subprocess.run(["pip", "uninstall", "hivetrain"])
    subprocess.run(["pip", "install", "-e", "."])
    
    # Stop the existing PM2 process (if running)
    subprocess.run(["pm2", "stop", "script"])
    
    # Start the script using PM2
    subprocess.run(["pm2", "start", "script.py", "--name", "script"])
    
    # Revert back to the original working directory
    os.chdir(original_dir)

def get_latest_commit_sha(repo_owner, repo_name, file_path):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data["sha"]
    else:
        print("Failed to fetch the latest commit SHA.")
        return None

def monitor_repo():
    repo_owner = "username"
    repo_name = "repo"
    file_path = "__init__.py"
    repo_dir = "repo"
    
    current_sha = get_latest_commit_sha(repo_owner, repo_name, file_path)
    
    while True:
        latest_sha = get_latest_commit_sha(repo_owner, repo_name, file_path)
        
        if current_sha is not None and latest_sha is not None and current_sha != latest_sha:
            print("__init__.py file updated. Running script...")
            run_script(repo_dir)
            current_sha = latest_sha
        
        # Sleep for a certain interval before checking again
        time.sleep(60)  # Check every 60 seconds, adjust as needed

# Start monitoring the repository
if __name__ == "__main__":
    monitor_repo()