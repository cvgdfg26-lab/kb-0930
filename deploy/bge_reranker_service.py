"""BGE Reranker HTTP 服务，按输入顺序返回 query 与每篇文档的相关性分数。

依赖：fastapi、uvicorn、FlagEmbedding、支持服务器 CUDA 的 PyTorch。
在本文件所在目录启动：
    python -m uvicorn bge_reranker_service:app --host 127.0.0.1 --port 6007 --workers 1

环境变量：
    BGE_RERANKER_LARGE：模型目录，默认 /usr-data/models/BAAI/bge-reranker-large。
    BGE_RERANKER_DEVICE：推理设备，默认 cuda:0；可设置为 cpu。
    BGE_RERANKER_FP16：是否启用半精度，默认 true；CPU 模式自动关闭。

请求示例：
    {"query": "RS12 怎么测交流电压？", "documents": ["交流电压测量方法", "电阻测量方法"]}
响应示例：
    {"scores": [0.87, 0.12]}

默认分数与现有 bge_rerank_util.py 的 compute_score 调用一致，为原始分数，
不保证处于 0～1 范围；请求中增加 "normalize": true 可返回 sigmoid 归一化分数。
"""

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, AsyncIterator, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictStr, field_validator

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker

logger = logging.getLogger("uvicorn.error")


class RerankRequest(BaseModel):
    """定义重排序请求：一个查询、一组文档，以及可选的分数归一化开关。"""

    query: StrictStr = Field(..., description="用于比较文档相关性的查询文本。")
    documents: List[StrictStr] = Field(
        ..., min_length=1, max_length=256, description="待打分文档，顺序会在响应中保持不变。"
    )
    normalize: bool = Field(False, description="为 true 时将原始分数映射到 0～1。")

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        """拒绝空查询及纯空白查询，保留有效查询的原始内容。"""
        if not query.strip():
            raise ValueError("query 不能为空或纯空白")
        return query

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, documents: List[str]) -> List[str]:
        """拒绝空文档，避免模型对无意义的 query-document 文本对推理。"""
        if any(not document.strip() for document in documents):
            raise ValueError("documents 中不能包含空字符串或纯空白文本")
        return documents


class RerankResponse(BaseModel):
    """定义响应：scores 中第 i 个分数对应 documents 中第 i 篇文档。"""

    scores: List[float]


def _patch_xlm_roberta_tokenizer() -> None:
    """兼容 transformers 5.x：为旧版 FlagEmbedding 补回已移除的分词器方法。"""
    from transformers import XLMRobertaTokenizer

    if hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        return

    def prepare_for_model(self, ids, pair_ids=None, **kwargs):
        """按现有项目的兼容实现组装查询与文档 token，并处理长度截断。"""
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        pair_ids = pair_ids or []

        if max_length and truncation in (True, "only_second"):
            special_tokens_count = 4 if pair_ids else 2
            max_pair_length = max_length - len(ids) - special_tokens_count
            if max_pair_length < len(pair_ids):
                pair_ids = pair_ids[: max(0, max_pair_length)]

        cls_token_id = getattr(self, "cls_token_id", None)
        sep_token_id = getattr(self, "sep_token_id", None)
        if cls_token_id is None:
            cls_token_id = 0
        if sep_token_id is None:
            sep_token_id = 2

        if pair_ids:
            input_ids = [cls_token_id] + ids + [sep_token_id, sep_token_id] + pair_ids + [sep_token_id]
        else:
            input_ids = [cls_token_id] + ids + [sep_token_id]
        return {"input_ids": input_ids}

    XLMRobertaTokenizer.prepare_for_model = prepare_for_model


def load_reranker_model() -> "FlagReranker":
    """检查本地模型和 GPU，按环境变量配置加载一个 FlagReranker 实例。"""
    model_path = Path(
        os.getenv("BGE_RERANKER_LARGE", "/usr-data/models/BAAI/bge-reranker-large")
    )
    if not model_path.is_dir():
        raise RuntimeError(f"模型目录不存在：{model_path}")

    device = os.getenv("BGE_RERANKER_DEVICE", "cuda:0").strip()
    use_fp16 = os.getenv("BGE_RERANKER_FP16", "true").lower() in ("true", "1", "yes")

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查驱动和 PyTorch，或设置 BGE_RERANKER_DEVICE=cpu")
    if device == "cpu":
        use_fp16 = False

    _patch_xlm_roberta_tokenizer()
    from FlagEmbedding import FlagReranker

    logger.info("加载 BGE Reranker：path=%s, device=%s, fp16=%s", model_path, device, use_fp16)
    return FlagReranker(
        model_name_or_path=str(model_path), device=device, use_fp16=use_fp16
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """服务启动时加载一次模型；关闭服务时释放模型引用。"""
    application.state.reranker_model = load_reranker_model()
    application.state.inference_lock = Lock()
    try:
        yield
    finally:
        application.state.reranker_model = None


app = FastAPI(title="BGE Reranker 服务", version="1.0.0", lifespan=lifespan)


def generate_rerank_scores(
    reranker_model: "FlagReranker", query: str, documents: List[str], normalize: bool
) -> RerankResponse:
    """按原顺序构建 query-document 文本对，并转换模型分数为 JSON 数字列表。"""
    sentence_pairs = [(query, document) for document in documents]
    if normalize:
        raw_scores = reranker_model.compute_score(
            sentence_pairs=sentence_pairs, normalize=True
        )
    else:
        # 与 knowledge/utils/bge_rerank_util.py 的默认原始分数保持一致。
        raw_scores = reranker_model.compute_score(sentence_pairs=sentence_pairs)

    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if not isinstance(raw_scores, (list, tuple)):
        raw_scores = [raw_scores]

    scores = [float(score) for score in raw_scores]
    if len(scores) != len(documents):
        raise RuntimeError("模型返回分数数量与文档数量不一致")
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError("模型返回了非有限分数")
    return RerankResponse(scores=scores)


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
    """处理 HTTP 请求，串行执行 GPU 推理并返回与文档逐项对应的分数。"""
    model = getattr(request.app.state, "reranker_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未就绪")

    try:
        # 避免多个请求同时推理造成 GPU 显存峰值叠加。
        with request.app.state.inference_lock:
            return generate_rerank_scores(
                model, payload.query, payload.documents, payload.normalize
            )
    except Exception as exc:
        logger.exception("BGE Reranker 打分失败")
        raise HTTPException(status_code=500, detail="重排序失败，请检查服务端日志") from exc
