from dataclasses import dataclass
import os
from typing import Dict


@dataclass
class BudgetConfig:
    max_llm_calls: int = 4
    max_total_tokens: int = 6500
    max_input_tokens: int = 5000
    max_output_tokens: int = 2000
    max_tool_calls: int = 8
    max_browser_calls: int = 3
    max_retries: int = 2
    timeout_seconds: float = 120.0


# Conservative defaults per research mode
DEFAULT_BUDGET_PROFILES: Dict[str, BudgetConfig] = {
    "fast": BudgetConfig(
        max_llm_calls=1,
        max_total_tokens=2500,
        max_input_tokens=2000,
        max_output_tokens=1000,
        max_tool_calls=2,
        max_browser_calls=1,
        max_retries=1,
        timeout_seconds=30.0,
    ),
    "normal": BudgetConfig(
        max_llm_calls=3,
        max_total_tokens=5000,
        max_input_tokens=4000,
        max_output_tokens=1500,
        max_tool_calls=5,
        max_browser_calls=2,
        max_retries=2,
        timeout_seconds=60.0,
    ),
    "deep": BudgetConfig(
        max_llm_calls=4,
        max_total_tokens=6500,
        max_input_tokens=5000,
        max_output_tokens=2000,
        max_tool_calls=8,
        max_browser_calls=3,
        max_retries=2,
        timeout_seconds=120.0,
    ),
    "academic": BudgetConfig(
        max_llm_calls=4,
        max_total_tokens=6500,
        max_input_tokens=5000,
        max_output_tokens=2000,
        max_tool_calls=8,
        max_browser_calls=2,
        max_retries=2,
        timeout_seconds=120.0,
    ),
}


def load_budget_config(mode: str = "deep") -> BudgetConfig:
    """
    Load BudgetConfig for a given mode with environment variable overrides.
    Environment variables can override general defaults or mode-specific values.
    Examples:
      BUDGET_MAX_LLM_CALLS=5
      BUDGET_FAST_MAX_TOTAL_TOKENS=3000
    """
    clean_mode = mode.lower().strip()
    base = DEFAULT_BUDGET_PROFILES.get(clean_mode, DEFAULT_BUDGET_PROFILES["deep"])

    def get_env_int(key_suffix: str, default_val: int) -> int:
        # Check mode-specific env var e.g. BUDGET_FAST_MAX_LLM_CALLS
        mode_key = f"BUDGET_{clean_mode.upper()}_{key_suffix}"
        if mode_key in os.environ:
            try:
                return int(os.environ[mode_key])
            except ValueError:
                pass
        # Check global env var e.g. BUDGET_MAX_LLM_CALLS
        global_key = f"BUDGET_{key_suffix}"
        if global_key in os.environ:
            try:
                return int(os.environ[global_key])
            except ValueError:
                pass
        return default_val

    def get_env_float(key_suffix: str, default_val: float) -> float:
        mode_key = f"BUDGET_{clean_mode.upper()}_{key_suffix}"
        if mode_key in os.environ:
            try:
                return float(os.environ[mode_key])
            except ValueError:
                pass
        global_key = f"BUDGET_{key_suffix}"
        if global_key in os.environ:
            try:
                return float(os.environ[global_key])
            except ValueError:
                pass
        return default_val

    return BudgetConfig(
        max_llm_calls=get_env_int("MAX_LLM_CALLS", base.max_llm_calls),
        max_total_tokens=get_env_int("MAX_TOTAL_TOKENS", base.max_total_tokens),
        max_input_tokens=get_env_int("MAX_INPUT_TOKENS", base.max_input_tokens),
        max_output_tokens=get_env_int("MAX_OUTPUT_TOKENS", base.max_output_tokens),
        max_tool_calls=get_env_int("MAX_TOOL_CALLS", base.max_tool_calls),
        max_browser_calls=get_env_int("MAX_BROWSER_CALLS", base.max_browser_calls),
        max_retries=get_env_int("MAX_RETRIES", base.max_retries),
        timeout_seconds=get_env_float("TIMEOUT_SECONDS", base.timeout_seconds),
    )
