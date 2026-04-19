import os
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from dotenv import load_dotenv

    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.is_file():
        load_dotenv(_env)
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


def get_model(model: str, max_tokens: int | None = None) -> OpenAIChatCompletionClient:
    kwargs: dict = {
        "model": model,
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return OpenAIChatCompletionClient(**kwargs)
