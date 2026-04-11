import logging

from agent import config

_LOGGER_CONFIGURED = False


def setup_logging():
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGER_CONFIGURED = True
