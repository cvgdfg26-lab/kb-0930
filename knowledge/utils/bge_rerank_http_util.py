"""通过 BGE Reranker HTTP 服务打分，兼容现有重排序节点的调用方式。

BGE_RERANKER_URL 可配置为服务根地址（如 http://host:6009），
也可配置为完整接口地址（如 http://host:6009/v1/rerank）。
"""

import logging
import math
import os
from threading import Lock
from typing import Optional
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_reranker_model: Optional["BGERerankerHTTPClient"] = None
_model_lock = Lock()
_MAX_DOCUMENTS_PER_REQUEST = 256  # 与远端 FastAPI 接口的单次请求上限一致。


class BGERerankerHTTPClient:
    """提供与原 FlagReranker 相同的 compute_score 调用入口。"""

    def __init__(self, url: str):
        """保存远端接口地址；创建客户端时不进行网络请求。"""
        self.url = url

    def compute_score(self, sentence_pairs, normalize=False):
        """对查询和文档配对打分，按输入顺序返回分数列表或单个分数。

        sentence_pairs 可为 [(query, document), ...]，也可为单组
        [query, document]。前者返回列表，后者返回浮点数，沿用原模型接口。
        normalize=True 时请求服务返回 0～1 分数；默认保留服务的原始分数。
        """
        single_pair = (
            isinstance(sentence_pairs, (list, tuple))
            and len(sentence_pairs) == 2
            and all(isinstance(text, str) for text in sentence_pairs)
        )
        if single_pair:
            pairs = [sentence_pairs]
        else:
            pairs = list(sentence_pairs)
        if not pairs:
            return []

        # 同一查询的文档合并为一次请求；下标用于恢复原始配对顺序。
        groups = {}
        for index, pair in enumerate(pairs):
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(text, str) and text.strip() for text in pair)
            ):
                raise ValueError("sentence_pairs 必须包含非空的 (query, document) 文本对")
            query, document = pair
            groups.setdefault(query, []).append((index, document))

        scores = [0.0] * len(pairs)
        for query, indexed_documents in groups.items():
            for start in range(0, len(indexed_documents), _MAX_DOCUMENTS_PER_REQUEST):
                batch = indexed_documents[start : start + _MAX_DOCUMENTS_PER_REQUEST]
                payload = {
                    "query": query,
                    "documents": [document for _, document in batch],
                }
                if normalize:
                    payload["normalize"] = True

                response = requests.post(self.url, json=payload, timeout=(5, 120))
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or not isinstance(result.get("scores"), list):
                    raise ValueError("Reranker 服务返回的 scores 不是列表")
                if len(result["scores"]) != len(batch):
                    raise ValueError("Reranker 服务返回的分数数量与文档数量不一致")

                for (index, _), raw_score in zip(batch, result["scores"]):
                    score = float(raw_score)
                    if not math.isfinite(score):
                        raise ValueError("Reranker 服务返回了非有限分数")
                    scores[index] = score

        return scores[0] if single_pair else scores


def get_reranker_model() -> Optional[BGERerankerHTTPClient]:
    """沿用原函数名，读取 BGE_RERANKER_URL 并返回复用的 HTTP 客户端。"""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    with _model_lock:
        if _reranker_model is not None:
            return _reranker_model

        url = os.getenv("BGE_RERANKER_URL", "").strip().rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            logger.error("BGE_RERANKER_URL 未配置或不是有效的 HTTP 地址")
            return None

        endpoint = url if url.endswith("/v1/rerank") else f"{url}/v1/rerank"
        _reranker_model = BGERerankerHTTPClient(endpoint)
        return _reranker_model
