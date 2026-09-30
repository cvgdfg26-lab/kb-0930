import os
import logging
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv
from transformers import XLMRobertaTokenizer

load_dotenv()
logger = logging.getLogger(__name__)
_reranker_model = None


def _patch_xlm_roberta_tokenizer() -> None:
    """
    FlagEmbedding still calls prepare_for_model, which transformers 5.x removed
    from XLMRobertaTokenizer. Add a small compatibility shim when needed.
    """
    if hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        return

    def prepare_for_model(self, ids, pair_ids=None, **kwargs):
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        pair_ids = pair_ids or []

        if max_length and truncation in (True, "only_second"):
            special_tokens_count = 4 if pair_ids else 2
            max_pair_length = max_length - len(ids) - special_tokens_count
            if max_pair_length < len(pair_ids):
                pair_ids = pair_ids[:max(0, max_pair_length)]

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


def get_reranker_model() -> FlagReranker:
    """
    获取 Reranker 模型实例（单例模式）
    """
    global _reranker_model
    try:
        if _reranker_model is None:
            _patch_xlm_roberta_tokenizer()

            model_path = os.getenv("BGE_RERANKER_LARGE")
            device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
            use_fp16 = os.getenv("BGE_RERANKER_FP16", "False").lower() == "true"

            logger.info(f"正在初始化 Reranker 模型，路径: {model_path}, 设备: {device}, fp16: {use_fp16}")

            _reranker_model = FlagReranker(
                model_name_or_path=model_path,
                device=device,
                use_fp16=use_fp16
            )

            logger.info("Reranker 模型初始化成功！")

        return _reranker_model

    except Exception as e:
        logger.error(f"初始化 Reranker 模型失败: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    reranker = get_reranker_model()

    # [0.1,0.2,0.3]=reranker.compute_score()
    # print(reranker)

    query = "1956年发生了哪一事件标志着人工智能这一学科的正式诞生？"

    documents = [
        "1950 年，艾伦·图灵发表了其具有里程碑意义的论文《计算机与智能》，将图灵测试作为衡量智能的标准，这一概念在人工智能的哲学研究和发展中具有基础性意义。",
        "1956 年的达特茅斯会议被认为是人工智能作为一个学科领域的诞生之地；在此会议上，约翰·麦卡锡等人创造了“人工智能”这一术语，并明确了其基本目标.",
        "1951 年，英国数学家兼计算机科学家艾伦·图灵也开发出了首个用于下棋的程序，这展示了人工智能在游戏策略方面的一个早期应用实例.",
        "1955 年，艾伦·纽厄尔、赫伯特·A·西蒙和克利夫·肖共同发明的“逻辑理论家”程序标志着首个真正的人工智能程序的诞生，该程序能够解决逻辑问题，类似于证明数学定理。."
    ]
    pairs = [(query, doc) for doc in documents]
    scores = reranker.compute_score(pairs)

    print(scores)
