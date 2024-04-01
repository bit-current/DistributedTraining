 # Monitor GitHub repository and run a script when the `__init__.py` file is updated

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Functions Description](#functions-description)
4. [monitor_repo() Function](#monitor-repo-function)
5. [Run Script (run_script()) Function](#run-script-function)
6. [Get Latest Commit SHA (get_latest_commit_sha()) Function](#get-latest-commit-sha-function)

## <a name="introduction"></a> Introduction
This Python script monitors a GitHub repository and runs a local script whenever the `__init__.py` file is updated. The script uses `os`, `subprocess`, `time`, and `requests` libraries.

## <a name="dependencies"></a> Dependencies
- python: >=3.7
- git: For cloning and pulling the repository
- pm2: To start and stop the local script

## <a name="functions-description"></a> Functions Description
### `run_script(repo_dir)`
This function checks if the given repository directory exists, removes it if it does, installs a package (hivetrain) using pip, stops an existing PM2 process, and starts the script using pm2.

### `get_latest_commit_sha(repo_owner, repo_name, file_path)`
This function retrieves the latest commit SHA for the given GitHub repository file path.

### `monitor_repo()`
This function checks if the GitHub repository's `__init__.py` file has been updated and runs the script if an update is detected. It also installs the package (hivetrain) and starts/stops the script using pm2 when necessary.

## <a name="monitor-repo-function"></a> monitor_repo() Function
The `monitor_repo()` function monitors a GitHub repository for changes to the `__init__.py` file, installs the package, and starts/stops the script accordingly. It checks every 60 seconds (adjustable).

## <a name="run-script-function"></a> Run Script (run_script()) Function
The `run_script()` function clones or pulls the latest version of a GitHub repository, installs the package using pip, stops an existing pm2 process and starts a new one. It also reverted back to the original working directory after completing the process.

## <a name="get-latest-commit-sha-function"></a> Get Latest Commit SHA (get_latest_commit_sha()) Function
The `get_latest_commit_sha()` function uses the GitHub API to get the latest commit SHA for a given file in a repository. If the request is successful, it returns the latest commit SHA; otherwise, it prints an error message and returns None.