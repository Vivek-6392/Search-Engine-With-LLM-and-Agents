import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from metrics.collector import MetricsCollector, get_current_metrics


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that records LLM token usage, latencies,
    context sizes, tool calls, and errors into a MetricsCollector.
    """

    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        stage: str = "agent_step",
    ):
        super().__init__()
        self._collector = collector
        self.stage = stage
        self._llm_starts: Dict[str, float] = {}
        self._tool_starts: Dict[str, float] = {}
        self._prompt_lengths: Dict[str, int] = {}

    @property
    def collector(self) -> Optional[MetricsCollector]:
        return self._collector or get_current_metrics()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        self._llm_starts[key] = time.perf_counter()
        total_chars = sum(len(p) for p in prompts)
        self._prompt_lengths[key] = total_chars

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        start_time = self._llm_starts.pop(key, None)
        latency = (time.perf_counter() - start_time) if start_time else 0.0
        prompt_chars = self._prompt_lengths.pop(key, 0)

        input_tokens = 0
        output_tokens = 0
        model_name = ""

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                input_tokens = token_usage.get(
                    "prompt_tokens", token_usage.get("input_tokens", 0)
                )
                output_tokens = token_usage.get(
                    "completion_tokens", token_usage.get("output_tokens", 0)
                )
            model_name = response.llm_output.get("model_name", "")

        # Inspect generations for additional metadata (Groq / OpenAI usage metadata)
        if (input_tokens == 0 or output_tokens == 0) and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        usage = getattr(msg, "usage_metadata", None)
                        if isinstance(usage, dict):
                            input_tokens = usage.get("input_tokens", input_tokens)
                            output_tokens = usage.get("output_tokens", output_tokens)
                        resp_meta = getattr(msg, "response_metadata", {})
                        if isinstance(resp_meta, dict):
                            tu = resp_meta.get("token_usage", {})
                            if isinstance(tu, dict):
                                input_tokens = tu.get("prompt_tokens", input_tokens)
                                output_tokens = tu.get(
                                    "completion_tokens", output_tokens
                                )
                            if not model_name:
                                model_name = resp_meta.get("model_name", "")

        col = self.collector
        if col:
            col.record_llm_call(
                stage=self.stage,
                prompt_chars=prompt_chars,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                latency_seconds=latency,
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        self._llm_starts.pop(key, None)
        self._prompt_lengths.pop(key, None)

        err_msg = str(error).lower()
        col = self.collector
        if col:
            if any(
                x in err_msg
                for x in [
                    "rate_limit",
                    "rate limit",
                    "429",
                    "tokens per minute",
                    "tpm",
                    "rpm",
                ]
            ):
                col.record_rate_limit(stage=self.stage, error_msg=str(error))

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        self._tool_starts[key] = time.perf_counter()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        start_time = self._tool_starts.pop(key, None)
        latency = (time.perf_counter() - start_time) if start_time else 0.0

        tool_name = name or kwargs.get("name") or "unknown_tool"
        col = self.collector
        if col:
            col.record_tool_call(tool_name=tool_name, duration=latency)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id else "default"
        start_time = self._tool_starts.pop(key, None)
        latency = (time.perf_counter() - start_time) if start_time else 0.0

        tool_name = name or kwargs.get("name") or "unknown_tool"
        col = self.collector
        if col:
            col.record_tool_call(tool_name=tool_name, duration=latency)
