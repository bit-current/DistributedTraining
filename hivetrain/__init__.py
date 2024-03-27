__version__ = "0.3.0"
version_split = __version__.split(".")
__spec_version__ = (100 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))

from . import btt_connector
from . import auth
