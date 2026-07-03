from importlib.metadata import PackageNotFoundError, version

from .grid import *
from .parameter_catalogue import *

try:
    __version__ = version("rayfish")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0"
