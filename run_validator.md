 # Neurons Validator Script Documentation

This Shell script is designed to manage and enforce the running of a Python script named `validator.py` in a Distributed Training project using Git for version synchronization and PM2 for process management. The script checks for updates on GitHub, reclones or enforces changes based on the branch, and starts or restarts the script with pm2.

## Initializing Variables

The script initializes several variables such as:

- `script`: The path to the Python script that needs to be run.
- `autoRunLoc`: The realpath of this shell script file.
- `proc_name`: The name of the process that will be managed by PM2.
- `args`: An array to store command-line arguments.
- `version_location`: The location of the `__init__.py` file in the repository where the version information is stored.
- `version`: The variable name used for storing the Python package version number.
- `repo`: The GitHub repository URL.
- `branch`: The desired branch for fetching updates from.
- `repo_url`: The base URL for cloning the Git repository.

## Checking pm2 Installation and Dependencies

Before proceeding, the script checks if PM2 is installed by using the `command -v`. If not, an error message is displayed and the script exits. It also checks if necessary dependencies, such as `jq`, are present.

## Helper Functions

The script includes several helper functions to perform specific tasks:

### `get_version_difference`

This function calculates the difference between two version numbers (in semver format) by splitting them into arrays and comparing each component individually.

### `read_version_value`

Reads the value of the Python package version from the specified file (in this case, the `__init__.py` file).

### `check_variable_value_on_github`

Fetches the current value of a GitHub file's content and checks if it contains the given variable name.

### `check_package_installed`

Checks whether the specified package is installed on the system using either `dpkg-query` for Linux or Homebrew for macOS.

### `strip_quotes`

Removes leading and trailing double quotes from a given string.

## Controlling the Clone and Enforce Workflow

The script provides two main functions to handle cloning/checking out the latest version of the repository (`check_and_clone`) and enforcing the main branch changes locally (`enforce_main`). These functions are responsible for updating the local environment with changes from GitHub.

## Managing Command-line Arguments

The script processes command-line arguments, keeping track of flags and their values. The `script` argument is mandatory; if it's not provided, an error message is displayed and the script exits.

## Enforcing Updates and Starting/Restarting the Python Script

Finally, the script checks whether packages are installed and performs Git updates (checking the GitHub value, cloning or enforcing changes) if necessary. Once all prerequisites are met, it creates a PM2 configuration file and starts or restarts the script as required.