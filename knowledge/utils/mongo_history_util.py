import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

from knowledge.core.paths import get_env_file_path

load_dotenv(get_env_file_path())


class HistoryMongoTool:
    """MongoDB-backed chat history helper."""

    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL")
        self.db_name = os.getenv("MONGO_DB_NAME")
        if not self.mongo_url or not self.db_name:
            raise ValueError("MONGO_URL or MONGO_DB_NAME is not configured")

        self.client = MongoClient(self.mongo_url)
        self.db = self.client[self.db_name]
        self.chat_message = self.db["chat_message"]
        self.chat_message.create_index([("session_id", 1), ("ts", -1)])
        logging.info("Successfully connected to MongoDB: %s", self.db_name)


def clear_history(session_id: str) -> int:
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info("Deleted %s messages for session %s", result.deleted_count, session_id)
        return result.deleted_count
    except Exception as e:
        logging.error("Error clearing history for session %s: %s", session_id, e)
        return 0


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: List[str] = None,
    image_urls: List[str] = None,
    message_id: str = None,
) -> str:
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "image_urls": image_urls or [],
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id

    result = mongo_tool.chat_message.insert_one(document)
    return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None},
                ],
            },
            {"$set": {"item_names": item_names}},
        )
        logging.info("Updated %s records to item_names: %s", result.modified_count, item_names)
        return result.modified_count
    except Exception as e:
        logging.error("Error updating history item_names: %s", e)
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    mongo_tool = get_history_mongo_tool()
    try:
        cursor = mongo_tool.chat_message.find({"session_id": session_id}).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logging.error("Error getting recent messages: %s", e)
        return []


_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool
