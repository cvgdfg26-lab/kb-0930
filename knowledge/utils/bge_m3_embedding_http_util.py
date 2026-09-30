"""通过 BGE-M3 HTTP 服务生成向量，兼容原有嵌入工具的调用方式。

BGE_M3_URL 可配置为服务根地址（如 http://host:6008），
也可配置为完整接口地址（如 http://host:6008/embeddings）。
"""

import logging
import math
import os
from threading import Lock
from typing import List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from scipy.sparse import csr_array

logger = logging.getLogger(__name__)
load_dotenv()

_model_lock = Lock()
_model: Optional["BGEM3HTTPEmbeddingFunction"] = None
_MAX_DOCUMENTS_PER_REQUEST = 256  # 与远端 FastAPI 接口的单次请求上限一致。


class BGEM3HTTPEmbeddingFunction:
    """模拟原 BGEM3EmbeddingFunction 的 encode_documents 返回结构。"""

    def __init__(self, url: str):
        """保存嵌入接口地址；不在创建客户端时发送网络请求。"""
        self.url = url

    def encode_documents(self, documents: List[str]):
        """批量调用 HTTP 接口，返回 NumPy 稠密向量和 SciPy CSR 稀疏矩阵。"""
        if not documents:
            raise ValueError("documents 不能为空")

        dense_vectors = []
        sparse_rows = []
        for start in range(0, len(documents), _MAX_DOCUMENTS_PER_REQUEST):
            batch = documents[start : start + _MAX_DOCUMENTS_PER_REQUEST]
            response = requests.post(
                self.url,
                json={"embedding_documents": batch},
                timeout=(5, 120),
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("BGE-M3 服务返回的 JSON 不是对象")

            batch_dense = result.get("dense")
            batch_sparse = result.get("sparse")
            if (
                not isinstance(batch_dense, list)
                or not isinstance(batch_sparse, list)
                or len(batch_dense) != len(batch)
                or len(batch_sparse) != len(batch)
            ):
                raise ValueError("BGE-M3 服务返回的向量数量与文档数量不一致")

            for dense, sparse in zip(batch_dense, batch_sparse):
                vector = np.asarray(dense, dtype=np.float32)
                if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                    raise ValueError("BGE-M3 服务返回的稠密向量无效")
                if not isinstance(sparse, dict):
                    raise ValueError("BGE-M3 服务返回的稀疏向量无效")

                sparse_vector = {}
                for token_id, weight in sparse.items():
                    index = int(token_id)  # JSON 对象的键是字符串，原工具返回整数 token ID。
                    value = float(weight)
                    if index < 0 or not math.isfinite(value):
                        raise ValueError("BGE-M3 服务返回的稀疏向量包含无效值")
                    sparse_vector[index] = value

                dense_vectors.append(vector)
                sparse_rows.append(sparse_vector)

        row_indices = []
        token_indices = []
        weights = []
        for row_index, sparse_vector in enumerate(sparse_rows):
            for token_id, weight in sparse_vector.items():
                row_indices.append(row_index)
                token_indices.append(token_id)
                weights.append(weight)

        vocabulary_size = max(token_indices, default=-1) + 1
        sparse_matrix = csr_array(
            (
                np.asarray(weights, dtype=np.float32),
                (row_indices, token_indices),
            ),
            shape=(len(documents), vocabulary_size),
        )
        return {"dense": dense_vectors, "sparse": sparse_matrix}


def get_beg_m3_embedding_model() -> Optional[BGEM3HTTPEmbeddingFunction]:
    """沿用原函数名，返回一个复用的 HTTP 客户端对象。"""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        url = os.getenv("BGE_M3_URL", "").strip().rstrip("/")
        if not url or not url.startswith(("http://", "https://")):
            logger.error("BGE_M3_URL 未配置或不是 HTTP 地址")
            return None
        endpoint = url if url.endswith("/embeddings") else f"{url}/embeddings"
        _model = BGEM3HTTPEmbeddingFunction(endpoint)
        return _model


def generate_hybrid_embeddings(
    embedding_model: BGEM3HTTPEmbeddingFunction, embedding_documents: List[str]
):
    """沿用原函数签名，返回 dense 列表及整数 token ID 到权重的稀疏字典。"""
    try:
        embedding_result = embedding_model.encode_documents(embedding_documents)
        csr_matrix = embedding_result["sparse"]
        sparse_vectors = []
        for index in range(len(embedding_documents)):
            start = csr_matrix.indptr[index]
            end = csr_matrix.indptr[index + 1]
            sparse_vectors.append(
                dict(
                    zip(
                        csr_matrix.indices[start:end].tolist(),
                        csr_matrix.data[start:end].tolist(),
                    )
                )
            )
        return {
            "dense": [vector.tolist() for vector in embedding_result["dense"]],
            "sparse": sparse_vectors,
        }
    except Exception:
        logger.exception("通过 HTTP 生成 BGE-M3 向量失败")
        return None
