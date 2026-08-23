"""Harbor MCP server exposes only the read-only allowlisted tools."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from grounded_agent.mcp_server import SERVER_NAME, build_harbor_mcp
from grounded_agent.tools import ALLOWED_TOOLS


def test_mcp_lists_only_allowlisted_read_tools() -> None:
    server = build_harbor_mcp()
    assert server.name == SERVER_NAME
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert names == set(ALLOWED_TOOLS)
    for tool in tools:
        assert "write" not in (tool.description or "").lower()


def test_mcp_query_knowledge_returns_labelled_nominations() -> None:
    server = build_harbor_mcp()
    blocks = asyncio.run(
        server.call_tool(
            "query_knowledge",
            {
                "question": (
                    "What must Harbor do when combining retrieval results "
                    "from different indexes?"
                )
            },
        )
    )
    payload = json.loads(blocks[0].text)
    assert payload["tool"] == "query_knowledge"
    assert payload["trust_profile"] == "durable_knowledge"
    paths = [item["source_path"] for item in payload["items"]]
    assert any(path.startswith("knowledge/") for path in paths)


def test_mcp_refuses_write_tool_name() -> None:
    server = build_harbor_mcp()
    with pytest.raises(ToolError, match="Unknown tool: write_index"):
        asyncio.run(server.call_tool("write_index", {"question": "x"}))
