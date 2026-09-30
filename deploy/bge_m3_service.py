"""BGE-M3 HTTP 服务，返回与现有工具类兼容的 dense 和 sparse 向量。

安装依赖（先确保 PyTorch 支持服务器的 CUDA）：
    pip install fastapi 'uvicorn[standard]' 'pymilvus[model]' FlagEmbedding

在 deploy 目录启动（单个进程只加载一份模型）：
    uvicorn bge_m3_service:app --host 127.0.0.1 --port 6007 --workers 1

可选环境变量：
    BGE_M3_PATH：模型目录，默认 /usr-data/models/BAAI/bge-m3。
    BGE_DEVICE：推理设备，默认 cuda:0；也可以设置为 cpu。
    BGE_FP16：是否启用半精度，默认 true；CPU 模式自动关闭。
    BGE_BATCH_SIZE：模型内部推理批大小，默认 16；显存不足时可调小。

POST /embeddings 请求示例：
    {"embedding_documents": ["RS12 万用表如何测量交流电压？", "如何判断电阻量程？"]}
响应仅包含 dense 和 sparse，列表顺序与输入文本一致。
JSON 对象的键只能是字符串；客户端写入 Milvus 前可恢复整数 token ID：
    sparse = [{int(k): v for k, v in row.items()} for row in result["sparse"]]
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, AsyncIterator, Dict, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictStr, field_validator

if TYPE_CHECKING:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

logger = logging.getLogger("uvicorn.error")


class EmbeddingsRequest(BaseModel):
    """定义请求正文，沿用现有工具类的 embedding_documents 参数名。"""

    embedding_documents: List[StrictStr] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="待编码文本列表，每次包含 1 至 256 条非空文本。",
    )

    @field_validator("embedding_documents")
    @classmethod
    def validate_documents(cls, documents: List[str]) -> List[str]:
        """拒绝空白文本，同时保留合法文本的原始内容和顺序。"""
        if any(not document.strip() for document in documents):
            raise ValueError("embedding_documents 中不能包含空字符串或纯空白文本")
        return documents


class EmbeddingsResponse(BaseModel):
    """定义 JSON 响应，稀疏向量的键为字符串形式的 token ID。"""

    dense: List[List[float]]
    sparse: List[Dict[str, float]]


def load_embedding_model() -> "BGEM3EmbeddingFunction":
    """从本地目录加载模型，并读取设备、精度和推理批大小配置。"""
    model_path = Path(os.getenv("BGE_M3_PATH", "/usr-data/models/BAAI/bge-m3"))
    if not model_path.is_dir():
        raise RuntimeError(f"模型目录不存在：{model_path}")

    device = os.getenv("BGE_DEVICE", "cuda:0").strip()
    use_fp16 = os.getenv("BGE_FP16", "true").lower() in ("true", "1", "yes")
    batch_size = int(os.getenv("BGE_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("BGE_BATCH_SIZE 必须是正整数")

    import torch
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 驱动和 PyTorch，或设置 BGE_DEVICE=cpu")
    if device == "cpu":
        use_fp16 = False

    logger.info("加载 BGE-M3 模型：path=%s, device=%s, fp16=%s", model_path, device, use_fp16)
    return BGEM3EmbeddingFunction(
        model_name=str(model_path),
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        normalize_embeddings=True,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """启动时加载一次模型；加载失败则终止启动，关闭时释放模型引用。"""
    application.state.embedding_model = load_embedding_model()
    application.state.inference_lock = Lock()
    try:
        yield
    finally:
        application.state.embedding_model = None


app = FastAPI(title="BGE-M3 向量服务", version="1.0.0", lifespan=lifespan)


def generate_hybrid_embeddings(
    embedding_model: "BGEM3EmbeddingFunction", embedding_documents: List[str]
) -> EmbeddingsResponse:
    """沿用现有工具类的编码和 CSR 逐行转换逻辑，生成可序列化的向量。"""
    embedding_result = embedding_model.encode_documents(embedding_documents)
    csr_array = embedding_result["sparse"]
    sparse_vectors = []
    for index in range(len(embedding_documents)):
        start = csr_array.indptr[index]
        end = csr_array.indptr[index + 1]
        sparse_vectors.append(
            {
                str(int(token_id)): float(weight)
                for token_id, weight in zip(csr_array.indices[start:end], csr_array.data[start:end])
            }
        )

    return EmbeddingsResponse(
        dense=[vector.tolist() for vector in embedding_result["dense"]],
        sparse=sparse_vectors,
    )


@app.post("/embeddings", response_model=EmbeddingsResponse)
def embeddings(payload: EmbeddingsRequest, request: Request) -> EmbeddingsResponse:
    """在线程池执行推理，以互斥锁限制单进程的 GPU 并发，并返回混合向量。"""
    model = getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未就绪")

    try:
        # 避免多个请求同时推理造成 GPU 显存峰值叠加。
        with request.app.state.inference_lock:
            return generate_hybrid_embeddings(model, payload.embedding_documents)
    except Exception as exc:
        logger.exception("BGE-M3 向量生成失败")
        raise HTTPException(status_code=500, detail="向量生成失败，请检查服务端日志") from exc
