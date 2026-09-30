"""Query routes."""

import asyncio
import mimetypes
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.core.deps import get_query_service
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import sse_generator
from knowledge.processor.query_process.base import setup_logging
from knowledge.utils.asyncio_util import install_asyncio_connection_reset_filter
from knowledge.utils.minio_util import get_minio_client
from minio.error import S3Error


def configure_event_loop_policy() -> None:
    """Avoid noisy Proactor socket shutdown errors on Windows SSE/MCP connections."""
    if sys.platform != "win32":
        return

    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is not None:
        asyncio.set_event_loop_policy(policy_cls())


configure_event_loop_policy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_asyncio_connection_reset_filter(asyncio.get_running_loop())
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_routes(app)
    return app


def register_routes(app: FastAPI):

    @app.get("/chat.html")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.get("/images/{bucket}/{object_name:path}")
    def get_image(bucket: str, object_name: str):
        """通过聊天服务读取 MinIO 图片，避免浏览器把服务器的 127.0.0.1 当作本机。"""
        # 仅允许读取知识库配置的桶，避免代理接口暴露其他对象。
        if bucket != os.getenv("MINIO_BUCKET_NAME") or not object_name:
            raise HTTPException(status_code=404, detail="图片不存在")
        image_type = mimetypes.guess_type(object_name)[0]
        if image_type not in ("image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"):
            raise HTTPException(status_code=404, detail="图片不存在")

        client = get_minio_client()
        if client is None:
            raise HTTPException(status_code=503, detail="图片存储不可用")
        try:
            response = client.get_object(bucket, object_name)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchBucket"):
                raise HTTPException(status_code=404, detail="图片不存在") from exc
            raise HTTPException(status_code=502, detail="读取图片失败") from exc

        def stream_image():
            """分块传输图片，并在请求结束时释放 MinIO 连接。"""
            try:
                yield from response.stream(amt=64 * 1024)
            finally:
                response.close()
                response.release_conn()

        return StreamingResponse(
            stream_image(),
            media_type=image_type,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ):
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        service.submit_query(task_id, request.is_stream)

        if request.is_stream:
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            return StreamSubmitResponse(
                message="Query submitted", session_id=session_id, task_id=task_id
            )

        service.run_query_graph(task_id, session_id, request.query, False)
        answer = service.get_answer(task_id)
        image_urls = service.get_image_urls(task_id)
        return QueryResponse(message="处理完成", session_id=session_id, answer=answer, image_urls=image_urls)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str, limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
