#!/bin/bash

# Initialize variables
script="neurons/miner.py"
proc_name="distributed_training_miner"
version_location="template/__init__.py"
repo_url="https://github.com/bit-current/DistributedTraining.git"  # Your repository URL
main_branch="main"


(# # Initialize variables
# script="neurons/miner.py"
# autoRunLoc=$(readlink -f "$0")
# proc_name="distributed_training_miner" 
# args=()
# version_location="./template/__init__.py"
# version="__version__"
# repo_url="https://github.com/bit-current/DistributedTraining.git/tree/dev_kb"

# old_args=$@)

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


get_version() {
    if [ $# -eq 0 ]; then
        grep '__version__' | cut -d '"' -f 2 | tr -d "'\""
    else
        grep '__version__' "$1" | cut -d '"' -f 2 | tr -d "'\""
    fi
}

check_and_clone() {
    cd ..
    rm -rf $(basename "$repo_url" .git)
    if git clone "$repo_url"; then
        echo "Successfully cloned repository."
        cd $(basename "$repo_url" .git) || exit
        # Additional setup after cloning, if necessary
        pip install -e .
        pm2 restart "$proc_name"  # Restart the PM2 process
    else
        echo "Failed to clone the repository. Please check the URL and your internet connection."
        exit 1
    fi
}

enforce_main() {
    git fetch origin "$main_branch"
    git reset --hard "origin/$main_branch"
    git clean -df
    # Additional commands after enforcing main, if necessary
    pip install -e .
    pm2 restart "$proc_name"  # Restart the PM2 process
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
if [ "$?" -eq 1 ]; then
# Start of the main script
    while true; do
        if [ -d .git ]; then
            local_version=$(get_version "$version_location")
            # Fetch remote changes without applying them
            git fetch origin "$main_branch"
            remote_version=$(git show "origin/$main_branch:$version_location" | get_version)

            current_branch=$(git rev-parse --abbrev-ref HEAD)

            if [ "$local_version" != "$remote_version" ]; then
                echo "Version mismatch detected. Local version: $local_version, Remote version: $remote_version."
                if [ "$current_branch" = "$main_branch" ]; then
                    # Case 3: On main branch, and versions differ. Delete local and reclone.
                    echo "On main branch with version mismatch. Recloning..."
                    check_and_clone
                else
                    # Case 2: On a different branch, enforce main.
                    echo "On branch $current_branch, enforcing main branch changes..."
                    enforce_main
                fi
            elif [ "$current_branch" != "$main_branch" ]; then
                # If on a different branch, but versions are the same, still enforce main
                echo "On branch $current_branch, different from main. Enforcing main branch changes..."
                enforce_main
            else
                echo "Local version is up-to-date with the remote main branch."
                # Your regular operations here, e.g., restarting services
                pm2 restart "$proc_name"
            fi
        else
            echo "This directory is not a Git repository."
            exit 1
        fi

        # Sleep for a predefined time before checking again
        sleep 300  # 20 minutes
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi





# while true; do
#     if [ -d "./.git" ]; then
#         if [ "$local_version" != "$remote_version" ]; then
#             # 2. Different branch handling
#             if [ "$current_branch" != "dev_kb" ]; then
#                 echo "On branch $current_branch, which differs from main. Resetting to match remote main..."
#                 git fetch origin dev_kb
#                 git reset --hard origin/dev_kb
#                 git clean -df
#             else
#                 # 3. Main branch handling
#                 echo "On main branch with updates available. Recloning repository..."
#                 echo "On main branch with updates available. Recloning repository..."
#                 echo "On main branch with updates available. Recloning repository..."
#                 cd ..
#                 rm -rf "$(basename $repo_url .git)"
#                 git clone "$repo_url"
#                 cd "$(basename $repo_url .git)"
#             fi
#             # Common steps after update
#             pip install -e .
#             pm2 restart $proc_name
#         else
#             # 4. No differences
#             echo "Local version matches remote main. No action required."
#             exit 1
#         fi
#         sleep 150       
#     done
# else
#     echo "Missing package 'jq'. Please install it for your system first."
# fi

  