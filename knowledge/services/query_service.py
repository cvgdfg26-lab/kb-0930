"""Query service."""

import logging
import uuid
from typing import Any, Dict, List

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.sse_util import SSEEvent, create_sse_queue, push_sse_event
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

logger = logging.getLogger(__name__)


class QueryService:
    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        """Submit query task and create an SSE queue in streaming mode."""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str, is_stream: bool):
        """Run the LangGraph query flow."""
        final_state: Dict[str, Any] = {}

        try:
            default_state = {
                "original_query": user_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }
            final_state = query_app.invoke(default_state) or {}
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"Query flow failed: {e}", exc_info=True)
            error_message = str(e)
            set_task_result(task_id, "error", error_message)
            update_task_status(task_id, TASK_STATUS_FAILED)
            final_state = {"error": error_message}
        else:
            answer = final_state.get("answer", "")
            image_urls = final_state.get("image_urls", []) or []
            if answer:
                set_task_result(task_id, "answer", answer)
            if image_urls:
                set_task_result(task_id, "image_urls", image_urls)
        finally:
            if is_stream:
                push_sse_event(task_id, SSEEvent.PROGRESS, {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })
                push_sse_event(task_id, SSEEvent.FINAL, {
                    "status": get_task_status(task_id),
                    "answer": final_state.get("answer", ""),
                    "error": final_state.get("error", ""),
                    "image_urls": final_state.get("image_urls", []) or [],
                })

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_image_urls(self, task_id: str) -> List[str]:
        return get_task_result(task_id, "image_urls", []) or []

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages

        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls": r.get("image_urls", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history

        return clear_history(session_id)
