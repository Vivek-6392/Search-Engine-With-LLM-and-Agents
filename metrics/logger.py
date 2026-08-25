from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """
    JSON formatter for structured application and metrics logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured event payload if attached
        if hasattr(record, "event_data") and isinstance(record.event_data, dict):
            log_entry.update(record.event_data)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


logger = logging.getLogger("deepsearchai")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if reloaded
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    logger.addHandler(handler)


def log_event(
    event: str,
    request_id: Optional[str] = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log a structured event with event metadata and request ID.
    """
    event_data = {
        "event": event,
        "request_id": request_id or "",
        **kwargs,
    }
    extra = {"event_data": event_data}
    msg = f"[{event}] {kwargs.get('message', '')}".strip()
    logger.log(level, msg, extra=extra)
