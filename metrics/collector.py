import contextvars
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class LLMCallContext:
    call_index: int
    stage: str
    chars: int
    estimated_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_seconds: float = 0.0


@dataclass
class NodeExecutionMetric:
    node_id: str
    task: str
    status: str = "PENDING"
    tool_used: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class RequestMetrics:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    research_mode: str = "deep"
    selected_model_provider: str = ""
    total_latency: float = 0.0
    planner_latency: float = 0.0
    dag_node_count: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    skipped_nodes: int = 0
    per_node_latency: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_call_count: int = 0
    tool_names: List[str] = field(default_factory=list)
    browser_calls: int = 0
    playwright_calls: int = 0
    http_fetch_calls: int = 0
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    rate_limit_errors: int = 0
    retry_details: List[Dict[str, Any]] = field(default_factory=list)
    final_synthesis_latency: float = 0.0
    context_size_per_llm_call: List[Dict[str, Any]] = field(default_factory=list)
    budget_exhausted: bool = False
    budget_status: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ContextVar for thread-local / async-local access to current active metrics collector
_current_metrics_collector: contextvars.ContextVar[Optional["MetricsCollector"]] = (
    contextvars.ContextVar("_current_metrics_collector", default=None)
)


def get_current_metrics() -> Optional["MetricsCollector"]:
    """Retrieve the current active MetricsCollector for this thread/context."""
    return _current_metrics_collector.get()


def set_current_metrics(collector: Optional["MetricsCollector"]):
    """Set the active MetricsCollector for this thread/context."""
    _current_metrics_collector.set(collector)


class MetricsCollector:
    """
    Thread-safe collector for request-level metrics and tracing in DeepSearchAI.
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        research_mode: str = "deep",
        model_provider: str = "",
    ):
        self.metrics = RequestMetrics(
            request_id=request_id or str(uuid.uuid4()),
            research_mode=research_mode,
            selected_model_provider=model_provider,
        )
        self._lock = threading.RLock()
        self._start_time: Optional[float] = None
        self._node_tracker: Dict[str, NodeExecutionMetric] = {}
        self._llm_call_counter: int = 0
        self._token = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and not self.metrics.error:
            self.metrics.error = str(exc_val)
        self.finish()

    def start(self):
        """Start tracking end-to-end request."""
        with self._lock:
            self._start_time = time.perf_counter()
            self._token = _current_metrics_collector.set(self)

    def finish(self):
        """Finalize end-to-end request and compute total latency."""
        with self._lock:
            if self._start_time is not None:
                self.metrics.total_latency = round(
                    time.perf_counter() - self._start_time, 4
                )
            if self._token:
                try:
                    _current_metrics_collector.reset(self._token)
                except Exception:
                    _current_metrics_collector.set(None)
                self._token = None
            else:
                _current_metrics_collector.set(None)

    def record_planner(self, latency: float, node_count: int):
        with self._lock:
            self.metrics.planner_latency = round(latency, 4)
            self.metrics.dag_node_count = node_count

    def record_node_start(self, node_id: str, task: str = ""):
        with self._lock:
            self._node_tracker[node_id] = NodeExecutionMetric(
                node_id=node_id,
                task=task,
                status="RUNNING",
                start_time=time.perf_counter(),
            )

    def record_node_end(
        self,
        node_id: str,
        status: str,
        tool_used: Optional[str] = None,
        error: Optional[str] = None,
    ):
        with self._lock:
            now = time.perf_counter()
            node_metric = self._node_tracker.get(node_id)
            if node_metric:
                node_metric.status = status
                node_metric.tool_used = tool_used
                node_metric.error = error
                node_metric.end_time = now
                if node_metric.start_time:
                    node_metric.latency_seconds = round(
                        now - node_metric.start_time, 4
                    )
                else:
                    node_metric.latency_seconds = 0.0
            else:
                node_metric = NodeExecutionMetric(
                    node_id=node_id,
                    task="",
                    status=status,
                    tool_used=tool_used,
                    error=error,
                    end_time=now,
                    latency_seconds=0.0,
                )
                self._node_tracker[node_id] = node_metric

            # Update aggregate node counts
            self._recalculate_node_aggregates()

    def _recalculate_node_aggregates(self):
        completed = 0
        failed = 0
        skipped = 0
        per_node = {}

        for nid, m in self._node_tracker.items():
            if m.status == "COMPLETED":
                completed += 1
            elif m.status == "FAILED":
                failed += 1
            elif m.status == "SKIPPED":
                skipped += 1

            per_node[nid] = {
                "latency_seconds": m.latency_seconds,
                "status": m.status,
                "tool_used": m.tool_used or "direct",
                "error": m.error,
                "task": m.task,
            }

        self.metrics.completed_nodes = completed
        self.metrics.failed_nodes = failed
        self.metrics.skipped_nodes = skipped
        self.metrics.per_node_latency = per_node
        if self.metrics.dag_node_count == 0:
            self.metrics.dag_node_count = len(self._node_tracker)

    def record_tool_call(self, tool_name: str, duration: float = 0.0):
        with self._lock:
            self.metrics.tool_call_count += 1
            self.metrics.tool_names.append(tool_name)

    def record_browser_call(self, fetch_type: str = "http", success: bool = True):
        with self._lock:
            self.metrics.browser_calls += 1
            if fetch_type == "http":
                self.metrics.http_fetch_calls += 1
            elif fetch_type == "playwright":
                self.metrics.playwright_calls += 1

    def record_llm_call(
        self,
        stage: str,
        prompt_chars: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        latency_seconds: float = 0.0,
    ):
        with self._lock:
            self._llm_call_counter += 1
            self.metrics.llm_call_count += 1

            # If token counts are not reported by API, estimate 1 token ~= 4 characters
            est_tokens = max(1, prompt_chars // 4)
            in_tok = input_tokens if input_tokens > 0 else est_tokens
            out_tok = output_tokens

            self.metrics.input_tokens += in_tok
            self.metrics.output_tokens += out_tok
            self.metrics.total_tokens += in_tok + out_tok

            call_ctx = {
                "call_index": self._llm_call_counter,
                "stage": stage,
                "chars": prompt_chars,
                "estimated_tokens": est_tokens,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "model": model or self.metrics.selected_model_provider,
                "latency_seconds": round(latency_seconds, 4),
            }
            self.metrics.context_size_per_llm_call.append(call_ctx)

    def record_retry(
        self,
        stage: str = "node",
        reason: str = "",
        error_type: str = "rate_limit",
        wait_duration_seconds: float = 0.0,
    ):
        with self._lock:
            self.metrics.retries += 1
            self.metrics.retry_details.append(
                {
                    "stage": stage,
                    "reason": reason[:200] if reason else "",
                    "error_type": error_type,
                    "wait_duration_seconds": round(wait_duration_seconds, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def record_rate_limit(self, stage: str = "node", error_msg: str = ""):
        with self._lock:
            self.metrics.rate_limit_errors += 1

    def record_synthesis(
        self,
        latency: float,
        context_chars: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ):
        with self._lock:
            self.metrics.final_synthesis_latency = round(latency, 4)
            self.record_llm_call(
                stage="final_synthesis",
                prompt_chars=context_chars,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                latency_seconds=latency,
            )

    def record_budget(self, budget_dict: Dict[str, Any], is_exhausted: bool = False):
        with self._lock:
            self.metrics.budget_status = budget_dict
            self.metrics.budget_exhausted = is_exhausted

    def get_metrics(self) -> RequestMetrics:
        with self._lock:
            return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return self.metrics.to_dict()

    def to_json(self, indent: int = 2) -> str:
        with self._lock:
            return self.metrics.to_json(indent=indent)
