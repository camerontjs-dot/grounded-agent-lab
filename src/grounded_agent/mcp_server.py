"""Read-only Harbor MCP server.

Exposes the same two query tools as the in-process allowlist. No write tools
are registered. This is a local fixture server, not a live MainFrame index.

The freeze lives in the FastMCP tool manager, which owns registration
(`add_tool`, `remove_tool`) and invocation (`call_tool`). It is not an instance
patch on the server. FastMCP binds its low-level `tools/list` and `tools/call`
handlers to the bound methods `self.list_tools` and `self.call_tool` at
construction, so a patch made later on the instance never reaches the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool, ToolManager
from mcp.types import Tool as MCPTool

from grounded_agent.tools import ALLOWED_TOOLS, FixtureToolClient, ToolDenied, ToolObservation

SERVER_NAME = "harbor-retrieval"


def _dump(observation: ToolObservation) -> dict:
    return observation.model_dump(mode="json")


class FrozenToolManager(ToolManager):
    """Tool registry that admits only ALLOWED_TOOLS and becomes read-only on freeze()."""

    def __init__(self, warn_on_duplicate_tools: bool = True) -> None:
        super().__init__(warn_on_duplicate_tools)
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        if self._frozen:
            return
        # Swap in a read-only view: `_tools[name] = ...` now raises TypeError.
        self._tools = MappingProxyType(dict(self._tools))
        self._frozen = True

    def add_tool(self, fn: Callable[..., Any], name: str | None = None, **options: Any) -> Tool:
        if self._frozen:
            raise ToolDenied("harbor MCP tool registry is frozen; write tools cannot be added")
        requested = name or getattr(fn, "__name__", "")
        if requested not in ALLOWED_TOOLS:
            raise ToolDenied(f"tool {requested!r} is not an allowlisted read-only tool")
        # Forward the SDK's options untouched; the set of options varies across mcp 1.x.
        tool = super().add_tool(fn, name=name, **options)
        if tool.name not in ALLOWED_TOOLS:
            # Drift guard: if the SDK derives a different name, drop the entry and refuse.
            self._tools.pop(tool.name, None)
            raise ToolDenied(f"tool {tool.name!r} is not an allowlisted read-only tool")
        return tool

    def remove_tool(self, name: str) -> None:
        if self._frozen:
            raise ToolDenied("harbor MCP tool registry is frozen; tools cannot be removed")
        super().remove_tool(name)

    def get_tool(self, name: str) -> Tool | None:
        if name not in ALLOWED_TOOLS:
            return None
        return super().get_tool(name)

    def list_tools(self) -> list[Tool]:
        return [tool for name, tool in self._tools.items() if name in ALLOWED_TOOLS]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        if name not in ALLOWED_TOOLS:
            raise ToolError(f"Unknown tool: {name}")
        return await super().call_tool(name, arguments, *args, **kwargs)


class HarborMCP(FastMCP):
    """FastMCP server whose tool registry is a FrozenToolManager."""

    def __init__(self, name: str = SERVER_NAME) -> None:
        super().__init__(name)
        manager = getattr(self, "_tool_manager", None)
        if not isinstance(manager, ToolManager):
            # Fail closed on SDK drift: never return a server with an unknown registry.
            raise RuntimeError(
                "FastMCP private _tool_manager is missing or not a ToolManager "
                f"({type(manager).__name__}); refusing to build the Harbor MCP server"
            )
        self.tool_registry = FrozenToolManager(
            warn_on_duplicate_tools=getattr(manager, "warn_on_duplicate_tools", True)
        )
        self._tool_manager = self.tool_registry

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        return [tool for tool in tools if tool.name in ALLOWED_TOOLS]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in ALLOWED_TOOLS:
            raise ToolError(f"Unknown tool: {name}")
        return await super().call_tool(name, arguments)


def build_harbor_mcp(client: FixtureToolClient | None = None) -> HarborMCP:
    backend = client or FixtureToolClient()
    server = HarborMCP(SERVER_NAME)

    @server.tool(
        name="query_knowledge",
        description="Read-only search of durable Harbor notes. Returns nominations, not proof.",
    )
    def query_knowledge(question: str) -> dict:
        return _dump(backend.invoke("query_knowledge", {"question": question}))

    @server.tool(
        name="query_projects",
        description="Read-only search of Harbor project-status notes. Nominations, not proof.",
    )
    def query_projects(question: str) -> dict:
        return _dump(backend.invoke("query_projects", {"question": question}))

    server.tool_registry.freeze()
    return server
