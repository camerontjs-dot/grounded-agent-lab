"""Read-only Harbor MCP server.

Exposes the same two query tools as the in-process allowlist. No write tools
are registered. This is a local fixture server, not a live MainFrame index.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from grounded_agent.tools import FixtureToolClient, ToolDenied, ToolObservation

SERVER_NAME = "harbor-retrieval"


def _dump(observation: ToolObservation) -> dict:
    return observation.model_dump(mode="json")


def build_harbor_mcp(client: FixtureToolClient | None = None) -> FastMCP:
    backend = client or FixtureToolClient()
    server = FastMCP(SERVER_NAME)

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

    def _frozen_add_tool(*args: object, **kwargs: object) -> None:
        raise ToolDenied("harbor MCP server is frozen; write tools cannot be added")

    server.add_tool = _frozen_add_tool  # type: ignore[method-assign]
    return server
