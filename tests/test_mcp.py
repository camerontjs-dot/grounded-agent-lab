"""Harbor MCP server exposes only the read-only allowlisted tools."""

from __future__ import annotations

import asyncio
import json

import mcp.server.fastmcp.server as sdk_server
import pytest
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool

from grounded_agent.mcp_server import SERVER_NAME, FrozenToolManager, HarborMCP, build_harbor_mcp
from grounded_agent.tools import ALLOWED_TOOLS, DENIED_WRITE_TOOLS, ToolDenied


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


def _write_index(question: str) -> str:
    return f"wrote {question}"


def _listed_names(server: FastMCP) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _low_level_listed(server: FastMCP) -> set[str]:
    handler = server._mcp_server.request_handlers[types.ListToolsRequest]
    result = asyncio.run(handler(types.ListToolsRequest(method="tools/list"))).root
    return {tool.name for tool in result.tools}


def _low_level_call(server: FastMCP, name: str, arguments: dict) -> types.CallToolResult:
    handler = server._mcp_server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    return asyncio.run(handler(request)).root


def test_build_installs_frozen_registry_with_only_read_tools() -> None:
    server = build_harbor_mcp()
    assert isinstance(server, HarborMCP)
    assert isinstance(server.tool_registry, FrozenToolManager)
    assert server._tool_manager is server.tool_registry
    assert server.tool_registry.frozen
    assert _listed_names(server) == set(ALLOWED_TOOLS)
    assert _low_level_listed(server) == set(ALLOWED_TOOLS)


def test_frozen_registry_refuses_write_registration_at_every_entry_point() -> None:
    server = build_harbor_mcp()
    with pytest.raises(ToolDenied, match="frozen"):
        server._tool_manager.add_tool(_write_index)
    with pytest.raises(ToolDenied, match="frozen"):
        server.add_tool(_write_index)
    with pytest.raises(ToolDenied, match="frozen"):
        server.tool(name="write_index")(_write_index)
    assert _listed_names(server) == set(ALLOWED_TOOLS)


def test_unfrozen_registry_admits_only_allowlisted_names() -> None:
    registry = FrozenToolManager()
    with pytest.raises(ToolDenied, match="not an allowlisted"):
        registry.add_tool(_write_index)
    with pytest.raises(ToolDenied, match="not an allowlisted"):
        registry.add_tool(lambda: None)

    def query_knowledge(question: str) -> str:
        return question

    registry.add_tool(query_knowledge)
    assert [tool.name for tool in registry.list_tools()] == ["query_knowledge"]
    assert registry.get_tool("write_index") is None


def test_manager_filters_a_registry_mutated_behind_its_back() -> None:
    registry = FrozenToolManager()
    registry._tools["write_index"] = Tool.from_function(_write_index, name="write_index")
    assert registry.get_tool("write_index") is None
    assert registry.list_tools() == []
    with pytest.raises(ToolError, match="Unknown tool: write_index"):
        asyncio.run(registry.call_tool("write_index", {"question": "x"}))


def test_direct_registry_mutation_and_removal_are_refused() -> None:
    server = build_harbor_mcp()
    registry = server.tool_registry
    write_tool = Tool.from_function(_write_index, name="write_index")
    with pytest.raises(TypeError):
        registry._tools["write_index"] = write_tool
    with pytest.raises(ToolDenied, match="frozen"):
        registry.remove_tool("query_knowledge")
    assert registry.get_tool("write_index") is None
    assert _listed_names(server) == set(ALLOWED_TOOLS)


def test_write_tool_never_lists_or_runs_through_call_tool() -> None:
    server = build_harbor_mcp()
    invoked: list[str] = []

    def write_index(question: str) -> str:
        invoked.append(question)
        return "wrote"

    with pytest.raises(ToolDenied, match="frozen"):
        server._tool_manager.add_tool(write_index)
    assert "write_index" not in _listed_names(server)
    assert "write_index" not in _low_level_listed(server)
    with pytest.raises(ToolError, match="Unknown tool: write_index"):
        asyncio.run(server.call_tool("write_index", {"question": "x"}))
    with pytest.raises(ToolError, match="Unknown tool: write_index"):
        asyncio.run(server._tool_manager.call_tool("write_index", {"question": "x"}))
    refused = _low_level_call(server, "write_index", {"question": "x"})
    assert refused.isError is True
    assert "Unknown tool: write_index" in refused.content[0].text
    assert invoked == []


@pytest.mark.parametrize("name", [*sorted(DENIED_WRITE_TOOLS), "not_a_tool"])
def test_unknown_and_write_names_fail_closed_on_both_paths(name: str) -> None:
    server = build_harbor_mcp()
    with pytest.raises(ToolError, match=f"Unknown tool: {name}"):
        asyncio.run(server.call_tool(name, {"question": "x"}))
    refused = _low_level_call(server, name, {"question": "x"})
    assert refused.isError is True
    assert f"Unknown tool: {name}" in refused.content[0].text


def test_allowed_tools_still_return_labelled_nominations_on_both_paths() -> None:
    server = build_harbor_mcp()
    question = "What is the next action for lantern-agent?"
    blocks = asyncio.run(server.call_tool("query_projects", {"question": question}))
    payload = json.loads(blocks[0].text)
    assert payload["tool"] == "query_projects"
    assert payload["trust_profile"] == "project_status"
    paths = [item["source_path"] for item in payload["items"]]
    assert paths and all(path.startswith("projects/") for path in paths)

    result = _low_level_call(server, "query_knowledge", {"question": question})
    assert result.isError is False
    knowledge = json.loads(result.content[0].text)
    assert knowledge["tool"] == "query_knowledge"
    assert knowledge["trust_profile"] == "durable_knowledge"


def test_server_overrides_refuse_write_even_when_manager_is_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = build_harbor_mcp()
    manager = server.tool_registry
    listed_before = manager.list_tools
    write_tool = Tool.from_function(_write_index, name="write_index")
    invoked: list[str] = []

    async def bypassed_call(name, arguments, context=None, convert_result=False):
        invoked.append(name)
        return ["write ran"]

    monkeypatch.setattr(manager, "list_tools", lambda: [*listed_before(), write_tool])
    monkeypatch.setattr(manager, "call_tool", bypassed_call)

    assert _listed_names(server) == set(ALLOWED_TOOLS)
    assert _low_level_listed(server) == set(ALLOWED_TOOLS)
    with pytest.raises(ToolError, match="Unknown tool: write_index"):
        asyncio.run(server.call_tool("write_index", {"question": "x"}))
    assert _low_level_call(server, "write_index", {"question": "x"}).isError is True
    assert invoked == []


def test_build_refuses_when_sdk_registry_is_not_a_tool_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdk_server, "ToolManager", lambda *args, **kwargs: object())
    with pytest.raises(RuntimeError, match="_tool_manager"):
        build_harbor_mcp()


def test_build_refuses_when_sdk_registry_attribute_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_init = FastMCP.__init__

    def drifted_init(self: FastMCP, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        del self._tool_manager

    monkeypatch.setattr(FastMCP, "__init__", drifted_init)
    with pytest.raises(RuntimeError, match="_tool_manager"):
        build_harbor_mcp()


def test_manager_drops_a_tool_the_sdk_names_outside_the_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Tool.from_function

    def renamed(cls: type[Tool], fn: object, **kwargs: object) -> Tool:
        return original(fn, **{**kwargs, "name": "write_index"})

    monkeypatch.setattr(Tool, "from_function", classmethod(renamed))
    registry = FrozenToolManager()

    def query_knowledge(question: str) -> str:
        return question

    with pytest.raises(ToolDenied, match="write_index"):
        registry.add_tool(query_knowledge)
    assert registry.list_tools() == []
    assert registry.get_tool("write_index") is None
