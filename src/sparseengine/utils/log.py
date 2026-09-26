import os
import sys
from loguru import logger


def quick_debug_print(something):
    return f'{something}'


logger.remove()


log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level
)


__all__ = ["logger", "log_once"]

_seen_messages = set()


def log_once(msg: str, level: str = 'INFO'):
    """Log each message once using a set of previously emitted messages."""
    if msg not in _seen_messages:

        logger.opt(depth=1).log(level.upper(), msg)
        _seen_messages.add(msg)
