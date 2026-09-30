import os

KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_FILE_PATH = os.path.join(KNOWLEDGE_ROOT, ".env")
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    return FRONT_PAGE_DIR


def get_env_file_path() -> str:
    return ENV_FILE_PATH
