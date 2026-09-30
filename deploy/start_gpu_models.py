"""在 GPU 容器中启动三个只监听本机的模型服务。"""

import os
import socket
import subprocess
from pathlib import Path


BASE = Path("/usr-data/apps/model_services")
PYTHON = Path("/root/autodl-tmp/venvs/model_services/bin/python")
LOG_DIR = BASE / "logs"


def port_is_open(port: int) -> bool:
    """检查本机端口是否已有服务监听，避免重复加载模型。"""
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def start(name: str, port: int, command: list[str], env: dict[str, str]) -> None:
    """把模型进程从当前终端分离，并将启动日志写入私有日志目录。"""
    if port_is_open(port):
        print(f"{name}: port {port} already listening")
        return
    LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=BASE / "0525",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"{name}: started pid={process.pid}, port={port}, log={log_path}")


def main() -> None:
    """读取私有 API 密钥并启动 BGE-M3、reranker 和 vLLM。"""
    secret_path = BASE / ".vllm_api_key"
    api_key = secret_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("vLLM API 密钥文件为空")

    common_env = os.environ.copy()
    common_env["CUDA_VISIBLE_DEVICES"] = "0"
    m3_env = {**common_env, "BGE_M3_PATH": "/usr-data/models/BAAI/bge-m3", "BGE_DEVICE": "cuda:0", "BGE_FP16": "true", "BGE_BATCH_SIZE": "8"}
    rerank_env = {**common_env, "BGE_RERANKER_LARGE": "/usr-data/models/BAAI/bge-reranker-large", "BGE_RERANKER_DEVICE": "cuda:0", "BGE_RERANKER_FP16": "true"}
    vllm_env = {**common_env, "VLLM_API_KEY": api_key}

    start("bge_m3", 6008, [str(PYTHON), "-m", "uvicorn", "bge_m3_service:app", "--host", "127.0.0.1", "--port", "6008", "--workers", "1"], m3_env)
    start("bge_reranker", 6009, [str(PYTHON), "-m", "uvicorn", "bge_reranker_service:app", "--host", "127.0.0.1", "--port", "6009", "--workers", "1"], rerank_env)
    # 24 GB 显卡同时运行三个模型时，显式限制 KV 缓存并关闭 CUDA Graph 预留。
    start("vllm", 6006, [str(PYTHON), "-m", "vllm.entrypoints.openai.api_server", "--model", "/usr-data/models/Qwen/Qwen3-8B", "--served-model-name", "qwen3-8b", "--host", "127.0.0.1", "--port", "6006", "--gpu-memory-utilization", "0.80", "--kv-cache-memory", "1073741824", "--enforce-eager", "--max-model-len", "4096"], vllm_env)


if __name__ == "__main__":
    main()
