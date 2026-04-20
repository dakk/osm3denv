import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def configure(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", stream=sys.stderr)
