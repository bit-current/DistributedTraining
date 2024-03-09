#!/bin/bash
# Inspired by https://github.com/surcyf123/smart-scrape/blob/main/run.sh

# Initialize variables
script="neurons/miner.py"
autoRunLoc=$(readlink -f "$0")
proc_name="distributed_training_miner" 
args=()
version_location="./template/__init__.py"
version="__version__"
repo_url="https://github.com/bit-current/DistributedTraining.git/tree/dev_kb"

old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)
    
    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

get_local_version() {
    grep "$version" $version_location | cut -d' ' -f3 | tr -d "'\""
}

# Define function to get remote main version
get_remote_version() {
    git fetch
    git show origin/main:$version_location | grep "$version" | cut -d' ' -f3 | tr -d "'\""
}


# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  # Check if the argument starts with a hyphen (flag)
  if [[ "$arg" == -* ]]; then
    # Check if the argument has a value
    if [[ $# -gt 1 && "$2" != -* ]]; then
          if [[ "$arg" == "--script" ]]; then
            script="$2";
            shift 2
        else
            # Add '=' sign between flag and value
            args+=("'$arg'");
            args+=("'$2'");
            shift 2
        fi
    else
      # Add '=True' for flags with no value
      args+=("'$arg'");
      shift
    fi
  else
    # Argument is not a flag, add it as it is
    args+=("'$arg '");
    shift
  fi
done

# Check if script argument was provided
if [[ -z "$script" ]]; then
    echo "The --script argument is required."
    exit 1
fi

branch=$(git branch --show-current)            # get current branch.
echo watching branch: $branch
echo pm2 process name: $proc_name

# Get the current version locally.
current_version=$(read_version_value)

# Check if script is already running with pm2
if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

# Run the Python script with the arguments using pm2
echo "Running $script with the following pm2 config:"

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma
joined_args=${joined_args%,}

# Create the pm2 config file
echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [$joined_args]
  }]
}" > app.config.js

# Print configuration to be used
cat app.config.js

pm2 start app.config.js

# Check if packages are installed.
check_package_installed "jq"

# Start main process
current_branch=$(git rev-parse --abbrev-ref HEAD)
local_version=$(get_local_version)
remote_version=$(get_remote_version)

while true; do
    if [ -d "./.git" ]; then
        if [ "$local_version" != "$remote_version" ]; then
            # 2. Different branch handling
            if [ "$current_branch" != "dev_kb" ]; then
                echo "On branch $current_branch, which differs from main. Resetting to match remote main..."
                git fetch origin dev_kb
                git reset --hard origin/dev_kb
                git clean -df
            else
                # 3. Main branch handling
                echo "On main branch with updates available. Recloning repository..."
                cd ..
                rm -rf "$(basename $repo_url .git)"
                git clone "$repo_url"
                cd "$(basename $repo_url .git)"
            fi
            # Common steps after update
            pip install -e .
            pm2 restart $proc_name
        else
            # 4. No differences
            echo "Local version matches remote main. No action required."
            exit 1
        fi
        sleep 150       
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi

  