
#!/bin/bash
# Inspired by https://github.com/surcyf123/smart-scrape/blob/main/run.sh

# Initialize variables
script="neurons/miner.py"
autoRunLoc=$(readlink -f "$0")
proc_name="distributed_training_miner" 
args=()
version_location="template/__init__.py"
version="__version__ "
repo="bit-current/DistributedTraining"
branch="main"
repo_url="https://github.com/$repo.git"


old_args=$@
echo "=====old_args"

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Returns the difference between 
# two versions as a numerical value.
get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    # Read each line in the file
    while IFS= read -r line; do
        # Check if the line contains the variable name
        if [[ "$line" == *"$version"* ]]; then
            # Extract the value of the variable
            local value=$(echo "$line" | awk -F '=' '{print $2}' | tr -d ' ')
            strip_quotes $value
            return 0
        fi
    done < "$version_location"

    echo ""
}


check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"
    local branch="$4"

    # Simplified URL construction to include the branch directly
    local url="https://api.github.com/repos/$repo/contents/$file_path?ref=$branch"
    
    # Fetch file content from GitHub and decode from Base64 in one go
    local variable_value=$(curl -s "$url" | jq -r '.content' | base64 --decode | grep "$variable_name" | cut -d '=' -f 2 | tr -d '[:space:]' | tr -d "'\"")

    if [[ -z "$variable_value" ]]; then
        echo "Error: Variable '$variable_name' not found in the file '$file_path' on branch '$branch'."
        return 1
    else
        echo "$variable_value"
    fi
}

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

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

# reclone and install packages

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



# enforce what's on main branch
enforce_main() {
    git stash
    git fetch origin "$branch"
    git reset --hard "origin/$branch"
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

local_branch=$(git branch --show-current)            # get current branch.
echo watching branch: $local_branch
echo pm2 process name: $proc_name

# Get the current version locally.
local_version=$(read_version_value)

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
    while true; do
        # First ensure that this is a git installation
        if [ -d "./.git" ]; then
            # Fetch remote changes without applying them
            git fetch origin "$branch"
            # check value on github remotely
            remote_version=$(check_variable_value_on_github "$repo" "$version_location" "$version" "$branch")

            if [ "$local_version" != "$remote_version" ]; then
            echo "Version mismatch detected. Local version: $local_version, Remote version: $remote_version."
            
                if [ "$local_branch" = "$branch" ]; then
                    # Case 3: On main branch, and versions differ. Delete local and reclone.
                    echo "On main branch with version mismatch. Recloning..."
                    check_and_clone
                else
                    # Case 2: On a different branch, enforce main.
                    echo "On branch $local_branch, enforcing main branch changes..."
                    enforce_main
                fi

                local_version=$(read_version_value)
                echo "Repository reset to the latest version."
                # Restart autorun script
                echo "Restarting script..."
                ./$(basename $0) $old_args && exit
            else
                echo "**Skipping update **"
                echo "$local_version is the same as or more than $remote_version. You are likely running locally."
            fi    
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/opentensor/validators and rerun this script."
        fi
        # wait for 3hrs and then check for changes again
        sleep 1800
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi


