 # Module Documentation

## Overview
This module provides functions related to managing the version of the software. It also imports necessary sub-modules for interacting with the BTT connector and handling authentication.

```python
__version__ = "0.3.0"
version_split = __version__.split(".")
__spec_version__ = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))
```

### Version Management
The version of the software is stored as a string in the `__version__` variable. The script then splits this string into its major, minor, and patch components using the `split()` method. These components are then used to calculate the semantic version number and store it in the `__spec_version__` variable.

The calculated semantic version number is obtained by multiplying each component with a factor (100 for major, 10 for minor, and 1 for patch) and summing the results. This calculation follows the standard semantic versioning format, where the first digit represents the major version, the second digit represents the minor version, and the third digit represents the patch version.

## Imported Sub-Modules
### btt_connector
The `btt_connector` sub-module contains functions for connecting to a BTT device or server using various communication protocols. It abstracts away the complexities of setting up connections, allowing users to interact with the devices in a simple and efficient way.

### auth
The `auth` sub-module provides functions for handling authentication procedures, such as login and logout. These functions ensure secure communication between the software and remote servers or devices by implementing proper encryption, token management, and error handling mechanisms.