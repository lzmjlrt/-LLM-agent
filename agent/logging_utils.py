import logging
import json
from datetime import UTC, datetime
from typing import Any

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


def log_event(event_type: str, **data: Any):
    """统一结构化业务事件日志。"""
    logger = logging.getLogger("agent.events")
    payload = {
        "event_type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
