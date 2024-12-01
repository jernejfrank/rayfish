from pathlib import Path

from .grid import *
from .parameter_catalogue import *

ROOT_DIRECTORY = Path(__file__).parent
with open(ROOT_DIRECTORY / "VERSION") as f:
    __version__ = f.read()
