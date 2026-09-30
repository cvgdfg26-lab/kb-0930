import asyncio
import json
from typing import Any, Dict, List, Tuple, Union

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class McpSearchNodeStreamable(BaseNode):
    name = "mcp_search_node_streamable"
    """
    使用 streamable HTTP 方式连接 MCP 服务端，调用百炼 WebSearch 工具，
    并将搜索结果整理为统一的 web_search_docs 结构。
    """

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_rewritten_query, _ = self._validate_query_inputs(state)

        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))
        if not mcp_result:
            return {}

        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        api_key = self.config.openai_api_key.strip()
        authorization = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"

        mcp_client = MCPServerStreamableHttp(
            name="通用搜索",
            params=MCPServerStreamableHttpParams(
                url=self.config.mcp_dashscope_base_url_streamable,
                headers={"Authorization": authorization},
                timeout=30,
                sse_read_timeout=60 * 5,
            ),
            cache_tools_list=True,
        )

        try:
            await mcp_client.connect()
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_rewritten_query, "count": 3},
            )

            if not execute_tool_result:
                return []

            if not getattr(execute_tool_result, "content", None):
                return []

            if not execute_tool_result.content[0]:
                return []

            text_content_text = execute_tool_result.content[0].text
            if not text_content_text:
                return []

            try:
                parsed_result: Dict[str, Any] = json.loads(text_content_text)
            except Exception:
                self.logger.exception("反序列化 MCP 结果失败")
                return []

            pages = parsed_result.get("pages", [])
            if not pages:
                return []

            search_result: List[Dict[str, Any]] = []
            for page in pages:
                snippet = page.get("snippet", "").strip()
                title = page.get("title", "").strip()
                url = page.get("url", "").strip()
                search_result.append({"snippet": snippet, "title": title, "url": url})

            return search_result
        finally:
            await mcp_client.cleanup()


if __name__ == "__main__":
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"],
    }

    mcp_search = McpSearchNodeStreamable()
    result = mcp_search.process(state)

    for r in result.get("web_search_docs", []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
