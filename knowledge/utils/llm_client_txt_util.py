"""文本大模型客户端：通过 VLLM_ENABLE 在 vLLM 与原有接口之间切换。"""

import logging
import os
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from knowledge.core.paths import get_env_file_path

# 明确读取项目的 .env，避免从其他工作目录启动时遗漏模型配置。
load_dotenv(get_env_file_path())

logger = logging.getLogger(__name__)
cache_llm_client = {}
_cache_lock = Lock()


def _normalize_api_base(value: str) -> str:
    """将服务根地址或完整聊天接口地址转换为 OpenAI 客户端需要的 /v1 基础地址。"""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("TEXT_LLM_OPENAI_API_BASE 必须是有效的 HTTP(S) 地址")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def get_llm_client(mode_name: str = None, temperature: float = 0.0, response_format: bool = False):
    """获取兼容旧调用方式的文本客户端，支持 invoke、stream 和 JSON 响应模式。

    VLLM_ENABLE=1 时使用 TEXT_LLM_* 配置；其他值沿用原有阿里云客户端。
    mode_name 显式传入时覆盖默认模型名，以保持旧函数的参数语义。
    """
    # 原有方案直接调用旧工具，视觉模型及其配置均不受文本模型切换影响。
    if os.getenv("VLLM_ENABLE", "0").strip() != "1":
        from knowledge.utils.llm_client_util import get_llm_client as get_legacy_client

        return get_legacy_client(
            mode_name=mode_name,
            temperature=temperature,
            response_format=response_format,
        )

    model_name = mode_name or os.getenv("TEXT_LLM_DEFAULT_MODEL", "").strip()
    api_key = os.getenv("TEXT_LLM_OPENAI_API_KEY", "").strip()
    api_base = os.getenv("TEXT_LLM_OPENAI_API_BASE", "").strip()
    if not model_name or not api_key or not api_base:
        logger.error("vLLM 文本模型配置不完整：请检查 TEXT_LLM_DEFAULT_MODEL、TEXT_LLM_OPENAI_API_KEY 和 TEXT_LLM_OPENAI_API_BASE")
        return None

    try:
        api_base = _normalize_api_base(api_base)
        cache_key = (model_name, api_base, api_key, temperature, response_format)
        with _cache_lock:
            if cache_key not in cache_llm_client:
                # Qwen3 关闭思考模式，确保普通回答与 JSON 模式返回可直接使用的文本。
                model_kwargs = {"response_format": {"type": "json_object"}} if response_format else {}
                cache_llm_client[cache_key] = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=api_key,
                    openai_api_base=api_base,
                    temperature=temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    model_kwargs=model_kwargs,
                )
            return cache_llm_client[cache_key]
    except Exception:
        logger.exception("vLLM 文本客户端创建失败")
        return None
