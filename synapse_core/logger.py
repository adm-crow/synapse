import logging
import sys
from typing import Optional


class CustomFormatter(logging.Formatter):
    white = "\x1b[37m"
    blue = "\x1b[34m"
    cyan = "\x1b[3;36m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"

    _fmt = "%(asctime)s - %(levelname)s : %(message)s"
    _datefmt = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: (white + _fmt + reset, _datefmt),
        logging.INFO: (blue + _fmt + reset, _datefmt),
        logging.WARNING: (yellow + _fmt + reset, _datefmt),
        logging.ERROR: (red + _fmt + reset, _datefmt),
        logging.CRITICAL: (bold_red + _fmt + reset, _datefmt),
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt, date_fmt = self.FORMATS.get(record.levelno, (self._fmt, self._datefmt))
        return logging.Formatter(log_fmt, date_fmt).format(record)


logger = logging.getLogger("synapse_core")
logger.setLevel(logging.INFO)

# Default: colored output to stdout (preserves existing verbose=True behaviour)
_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(CustomFormatter())
logger.addHandler(_default_handler)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure synapse_core logging output.

    By default synapse_core writes colored INFO messages to stdout.
    Call this function to change the level, silence output, or add a log file.

    Args:
        level:    Logging level, e.g. ``logging.DEBUG`` or ``logging.WARNING``.
                  Pass ``logging.CRITICAL`` to silence all output.
        log_file: Optional path to a log file. Messages are written there in
                  addition to stdout (plain text, no ANSI colour codes).

    Example::

        import logging
        import synapse_core

        # Colored stdout + persistent log file
        synapse_core.setup_logging(level=logging.DEBUG, log_file="ingest.log")

        # Silence all synapse_core output
        synapse_core.setup_logging(level=logging.CRITICAL)
    """
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler with colours
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Optional plain-text file handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s : %(message)s",
            "%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
