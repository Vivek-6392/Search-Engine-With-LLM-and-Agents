from dataclasses import dataclass
from enum import Enum
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class RateLimitType(str, Enum):
    RATE_LIMIT_429 = "rate_limit_429"
    CONTEXT_LENGTH_413 = "context_length_413"
    SERVICE_UNAVAILABLE_503 = "service_unavailable_503"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class RateLimitInfo:
    error_type: RateLimitType
    retry_after: Optional[float]
    message: str
    is_retryable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "retry_after": self.retry_after,
            "message": self.message[:200] if self.message else "",
            "is_retryable": self.is_retryable,
        }


def parse_retry_after(error: Any) -> Optional[float]:
    """
    Parse retry-after duration (in seconds) from Groq/OpenAI error message or response headers.
    Examples:
    - "Please try again in 2.34s" -> 2.34
    - "Please try again in 450ms" -> 0.45
    - "Retry-After: 4" -> 4.0
    - "retry after 3.5 seconds" -> 3.5
    """
    if error is None:
        return None

    # Check response headers if available on error object
    headers = getattr(error, "headers", None) or getattr(getattr(error, "response", None), "headers", None)
    if headers and isinstance(headers, dict):
        for k, v in headers.items():
            if k.lower() in ("retry-after", "x-ratelimit-reset", "retry-after-ms"):
                try:
                    val = float(v)
                    if "ms" in k.lower() or val > 1000:
                        return round(val / 1000.0, 3)
                    return round(val, 3)
                except ValueError:
                    pass

    # Parse from error message string
    err_str = str(error).lower()

    # Match "in X.XXs" or "in XXs"
    sec_match = re.search(r"(?:in|after)\s*([\d\.]+)\s*(?:s|sec|seconds?)", err_str)
    if sec_match:
        try:
            return round(float(sec_match.group(1)), 3)
        except ValueError:
            pass

    # Match "in XXXms"
    ms_match = re.search(r"(?:in|after)\s*([\d\.]+)\s*ms", err_str)
    if ms_match:
        try:
            return round(float(ms_match.group(1)) / 1000.0, 3)
        except ValueError:
            pass

    # Match "retry-after:\s*([\d\.]+)"
    header_match = re.search(r"retry-after[:\s]+([\d\.]+)", err_str)
    if header_match:
        try:
            return round(float(header_match.group(1)), 3)
        except ValueError:
            pass

    return None


def classify_llm_error(error: Exception) -> RateLimitInfo:
    """
    Inspect exception and classify into standard RateLimitInfo.
    """
    if error is None:
        return RateLimitInfo(
            error_type=RateLimitType.UNKNOWN,
            retry_after=None,
            message="",
            is_retryable=False,
        )

    msg = str(error)
    msg_lower = msg.lower()
    retry_after = parse_retry_after(error)

    # 1. Context length / Token limits exceeded (413)
    if any(
        cl in msg_lower
        for cl in [
            "context_length_exceeded",
            "context length",
            "maximum context length",
            "413",
            "request too large",
            "too many tokens",
            "prompt is too long",
        ]
    ):
        return RateLimitInfo(
            error_type=RateLimitType.CONTEXT_LENGTH_413,
            retry_after=retry_after,
            message=msg,
            is_retryable=True,  # Retryable only if prompt is compressed
        )

    # 2. Rate Limit (429 / TPM / RPM / RPD)
    if any(
        rl in msg_lower
        for rl in [
            "rate_limit",
            "rate limit",
            "429",
            "tpm",
            "rpm",
            "rpd",
            "tokens per minute",
            "requests per minute",
            "quota exceeded",
        ]
    ):
        return RateLimitInfo(
            error_type=RateLimitType.RATE_LIMIT_429,
            retry_after=retry_after,
            message=msg,
            is_retryable=True,
        )

    # 3. Service Unavailable / Overloaded (503 / 502)
    if any(
        su in msg_lower
        for su in [
            "503",
            "502",
            "service unavailable",
            "overloaded",
            "server error",
            "bad gateway",
        ]
    ):
        return RateLimitInfo(
            error_type=RateLimitType.SERVICE_UNAVAILABLE_503,
            retry_after=retry_after,
            message=msg,
            is_retryable=True,
        )

    # 4. Timeout / Network error
    if any(
        to in msg_lower
        for to in [
            "timeout",
            "timed out",
            "connection error",
            "connect error",
            "read timeout",
        ]
    ):
        return RateLimitInfo(
            error_type=RateLimitType.TIMEOUT,
            retry_after=retry_after,
            message=msg,
            is_retryable=True,
        )

    # 5. Non-retryable errors (e.g. 401 Unauthorized, 400 Bad Request, SyntaxError)
    return RateLimitInfo(
        error_type=RateLimitType.UNKNOWN,
        retry_after=None,
        message=msg,
        is_retryable=False,
    )


def calculate_backoff_delay(
    attempt: int,
    retry_after: Optional[float] = None,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
) -> float:
    """
    Calculate backoff delay with exponential scaling and full randomized jitter.
    Honors server-provided Retry-After when available.
    """
    if retry_after is not None and 0.0 < retry_after <= max_delay:
        # Add slight jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.4)
        return min(max_delay, retry_after + jitter)

    # Exponential backoff: base_delay * 2^attempt
    exp_delay = base_delay * (2 ** attempt)
    capped_delay = min(max_delay, exp_delay)
    jitter = random.uniform(0.05, 0.35)
    return min(max_delay, capped_delay + jitter)


def compress_prompt_for_context_limit(prompt: str) -> str:
    """
    Aggressively compress / truncate prompt context when encountering 413 context-length overflow.
    """
    if not prompt or len(prompt) < 500:
        return prompt

    lines = prompt.splitlines()
    if len(lines) <= 6:
        # Halve the characters
        return prompt[: len(prompt) // 2] + "\n...[context compressed for token limits]"

    # Retain prompt header and footer, but drop intermediate verbose sections
    head = lines[: len(lines) // 3]
    tail = lines[-len(lines) // 3 :]
    return "\n".join(head + ["...[intermediate context trimmed for context limit]..."] + tail)
