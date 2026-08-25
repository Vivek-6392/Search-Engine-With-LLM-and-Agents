import contextvars
from dataclasses import asdict
import threading
import time
from typing import Any, Dict, Optional

from budget.config import BudgetConfig, load_budget_config


_current_budget: contextvars.ContextVar[Optional["QueryBudget"]] = contextvars.ContextVar(
    "_current_budget", default=None
)


def get_current_budget() -> Optional["QueryBudget"]:
    """Get active QueryBudget from context."""
    return _current_budget.get()


def set_current_budget(budget: Optional["QueryBudget"]):
    """Set active QueryBudget in context."""
    _current_budget.set(budget)


class QueryBudget:
    """
    Global, thread-safe query resource and budget tracker.
    Coordinates limits across Planner, Executor, Tools, Browser, Synthesis, and Retries.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        mode: str = "deep",
    ):
        self.mode = mode.lower().strip()
        self.config = config or load_budget_config(self.mode)
        self._lock = threading.RLock()
        self._start_time: float = time.perf_counter()
        self._token = None

        # Consumed usage
        self.llm_calls_used: int = 0
        self.input_tokens_used: int = 0
        self.output_tokens_used: int = 0
        self.total_tokens_used: int = 0
        self.tool_calls_used: int = 0
        self.browser_calls_used: int = 0
        self.retries_used: int = 0

        # Active reservations
        self.llm_calls_reserved: int = 0
        self.tokens_reserved: int = 0
        self.tool_calls_reserved: int = 0
        self.browser_calls_reserved: int = 0
        self.retries_reserved: int = 0

        # Exhaustion state
        self._exhaustion_reasons: list[str] = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def start(self):
        """Start budget timer and activate context."""
        with self._lock:
            self._start_time = time.perf_counter()
            self._token = _current_budget.set(self)

    def finish(self):
        """Reset context on completion."""
        with self._lock:
            if self._token:
                try:
                    _current_budget.reset(self._token)
                except Exception:
                    _current_budget.set(None)
                self._token = None
            else:
                _current_budget.set(None)

    def remaining_time(self) -> float:
        """Remaining wall-clock seconds before request timeout."""
        with self._lock:
            elapsed = time.perf_counter() - self._start_time
            return max(0.0, self.config.timeout_seconds - elapsed)

    def remaining_tokens(self) -> int:
        """Remaining total tokens available (unconsumed and unreserved)."""
        with self._lock:
            allocated = self.total_tokens_used + self.tokens_reserved
            return max(0, self.config.max_total_tokens - allocated)

    def can_call_llm(self, estimated_tokens: int = 0) -> bool:
        """Check if an LLM invocation is permitted within budget."""
        with self._lock:
            if self.remaining_time() <= 0.0:
                self._check_and_record_exhaustion("timeout")
                return False

            if (self.llm_calls_used + self.llm_calls_reserved) >= self.config.max_llm_calls:
                self._check_and_record_exhaustion("max_llm_calls")
                return False

            if estimated_tokens > 0 and self.remaining_tokens() < estimated_tokens:
                self._check_and_record_exhaustion("max_total_tokens")
                return False

            if self.remaining_tokens() <= 0:
                self._check_and_record_exhaustion("max_total_tokens")
                return False

            return True

    def reserve(
        self,
        resource: str = "llm",
        count: int = 1,
        tokens: int = 0,
    ) -> bool:
        """
        Thread-safe reservation of budget slots.
        Returns True if reservation succeeds, False if budget would be exceeded.
        """
        with self._lock:
            if self.remaining_time() <= 0.0:
                self._check_and_record_exhaustion("timeout")
                return False

            if resource == "llm":
                if (self.llm_calls_used + self.llm_calls_reserved + count) > self.config.max_llm_calls:
                    self._check_and_record_exhaustion("max_llm_calls")
                    return False
                if tokens > 0 and (self.total_tokens_used + self.tokens_reserved + tokens) > self.config.max_total_tokens:
                    self._check_and_record_exhaustion("max_total_tokens")
                    return False
                self.llm_calls_reserved += count
                self.tokens_reserved += tokens
                return True

            elif resource == "tool":
                if (self.tool_calls_used + self.tool_calls_reserved + count) > self.config.max_tool_calls:
                    self._check_and_record_exhaustion("max_tool_calls")
                    return False
                self.tool_calls_reserved += count
                return True

            elif resource == "browser":
                if (self.browser_calls_used + self.browser_calls_reserved + count) > self.config.max_browser_calls:
                    self._check_and_record_exhaustion("max_browser_calls")
                    return False
                self.browser_calls_reserved += count
                return True

            elif resource == "retry":
                if (self.retries_used + self.retries_reserved + count) > self.config.max_retries:
                    self._check_and_record_exhaustion("max_retries")
                    return False
                self.retries_reserved += count
                return True

            return False

    def record_usage(
        self,
        llm_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
        browser_calls: int = 0,
        retries: int = 0,
        release_reservation: bool = True,
    ):
        """Thread-safe recording of actual consumed usage."""
        with self._lock:
            if release_reservation:
                if llm_calls > 0:
                    self.llm_calls_reserved = max(0, self.llm_calls_reserved - llm_calls)
                if (input_tokens + output_tokens) > 0:
                    self.tokens_reserved = max(0, self.tokens_reserved - (input_tokens + output_tokens))
                if tool_calls > 0:
                    self.tool_calls_reserved = max(0, self.tool_calls_reserved - tool_calls)
                if browser_calls > 0:
                    self.browser_calls_reserved = max(0, self.browser_calls_reserved - browser_calls)
                if retries > 0:
                    self.retries_reserved = max(0, self.retries_reserved - retries)

            self.llm_calls_used += llm_calls
            self.input_tokens_used += input_tokens
            self.output_tokens_used += output_tokens
            self.total_tokens_used += input_tokens + output_tokens
            self.tool_calls_used += tool_calls
            self.browser_calls_used += browser_calls
            self.retries_used += retries

            self._check_exhaustion_conditions()

    def _check_and_record_exhaustion(self, reason: str):
        if reason not in self._exhaustion_reasons:
            self._exhaustion_reasons.append(reason)

    def _check_exhaustion_conditions(self):
        if self.llm_calls_used >= self.config.max_llm_calls:
            self._check_and_record_exhaustion("max_llm_calls")
        if self.total_tokens_used >= self.config.max_total_tokens:
            self._check_and_record_exhaustion("max_total_tokens")
        if self.tool_calls_used >= self.config.max_tool_calls:
            self._check_and_record_exhaustion("max_tool_calls")
        if self.browser_calls_used >= self.config.max_browser_calls:
            self._check_and_record_exhaustion("max_browser_calls")
        if self.retries_used >= self.config.max_retries:
            self._check_and_record_exhaustion("max_retries")
        if self.remaining_time() <= 0.0:
            self._check_and_record_exhaustion("timeout")

    def exhausted(self) -> bool:
        """Returns True if any hard limit has been exceeded."""
        with self._lock:
            self._check_exhaustion_conditions()
            return len(self._exhaustion_reasons) > 0

    def should_skip_optional_work(self) -> bool:
        """
        Returns True when budget is depleted or running low, ensuring remaining
        headroom is reserved strictly for final report synthesis.
        """
        with self._lock:
            if self.exhausted():
                return True

            # If remaining LLM headroom is <= 1, reserve it for final synthesis!
            available_llm = self.config.max_llm_calls - (self.llm_calls_used + self.llm_calls_reserved)
            if available_llm <= 1:
                return True

            # If remaining token headroom is critically low (< 1000 tokens)
            if self.remaining_tokens() < 1000:
                return True

            # If remaining time is under 5 seconds
            if self.remaining_time() < 5.0:
                return True

            return False

    def get_status(self) -> Dict[str, Any]:
        """Summary of budget status, usage, and exhaustion reasons."""
        with self._lock:
            self._check_exhaustion_conditions()
            is_exhausted = len(self._exhaustion_reasons) > 0
            return {
                "mode": self.mode,
                "exhausted": is_exhausted,
                "exhaustion_reasons": list(self._exhaustion_reasons),
                "remaining_time_seconds": round(self.remaining_time(), 2),
                "remaining_tokens": self.remaining_tokens(),
                "llm_calls": f"{self.llm_calls_used}/{self.config.max_llm_calls}",
                "total_tokens": f"{self.total_tokens_used}/{self.config.max_total_tokens}",
                "input_tokens": f"{self.input_tokens_used}/{self.config.max_input_tokens}",
                "output_tokens": f"{self.output_tokens_used}/{self.config.max_output_tokens}",
                "tool_calls": f"{self.tool_calls_used}/{self.config.max_tool_calls}",
                "browser_calls": f"{self.browser_calls_used}/{self.config.max_browser_calls}",
                "retries": f"{self.retries_used}/{self.config.max_retries}",
                "config": asdict(self.config),
            }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_status()
