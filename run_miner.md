 # Script Documentation

This script is designed to automate the process of updating and running a Python script called `miner.py` using Git, pm2, and various utilities like jq and base64. The script also ensures that the required packages are installed and the Python version is up-to-date.

## Prerequisites

Before executing this script, make sure you have the following dependencies installed:

1. Git
2. pm2
3. jq (optional but recommended for checking remote versions)
4. Python 3 and pip for running the script

## Script Logic

The script begins by initializing various variables such as the path to the Python script, the name of the process that will be run using pm2, arguments for cloning the repository and installing packages, and locations of important files.

Next, it checks if the required dependency `pm2` is installed. If not, the script exits with an error message.

### get_version_difference Function

This function calculates the difference in version numbers between two strings provided as arguments (in the format 'vX.Y.Z'). It returns the difference as a numerical value.

### read_version_value Function

This function reads the value of the `__version__` variable from the specified file and stores it in the local variable `local_version`.

### check_variable_value_on_github Function

This function fetches the value of a given variable from a specific GitHub repository's file on the main branch and returns it.

### check_package_installed Function

This function checks whether a specified package (such as jq) is installed based on the operating system.

### strip_quotes Function

This function removes leading and trailing quotes from a given string.

## Main Logic

The script then performs the following tasks:

1. It clones the required repository if it's not already available.
2. Installs required packages using pip.
3. Starts the Python script with pm2.
4. Checks for updated versions of the script on GitHub and enforces the latest version if necessary.
5. Continuously checks for updated versions every 3 hours until the local version matches or exceeds the remote version. If the local branch is not main, it enforces the main branch's changes instead. This process repeats indefinitely.

In summary, this script automates updating and running a Python script while ensuring that all dependencies are met and the latest versions of the code and packages are used.