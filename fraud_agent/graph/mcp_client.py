"""Synchronous wrapper around the TigerGraph MCP server (stdio).

The agent's graph tools are executed as MCP tool calls against `tigergraph-mcp`, so the
LLM-facing layer never talks to TigerGraph REST directly. One long-lived MCP session runs
on a background event loop; `call()` is thread-safe and blocking.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPError(RuntimeError):
    pass


def _server_command() -> str:
    exe = shutil.which("tigergraph-mcp")
    if exe:
        return exe
    cand = Path(sys.executable).parent / ("tigergraph-mcp.exe" if os.name == "nt" else "tigergraph-mcp")
    if cand.exists():
        return str(cand)
    raise MCPError("tigergraph-mcp not found; pip install tigergraph-mcp")


class TigerGraphMCP:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._err: BaseException | None = None
        self.calls: list[dict] = []          # audit trail of every MCP call
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(60):
            raise MCPError("timed out starting tigergraph-mcp")
        if self._err:
            raise MCPError(f"tigergraph-mcp failed to start: {self._err}")

    # ---------------------------------------------------------------- lifecycle
    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        env = {k: v for k, v in os.environ.items() if k.startswith("TG_")}
        env.update({"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        params = StdioServerParameters(command=_server_command(), args=[], env=env)
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    self._session = s
                    self.tool_names = [t.name for t in (await s.list_tools()).tools]
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as e:  # noqa: BLE001
            self._err = e
            self._ready.set()

    def close(self):
        if self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)

    # ---------------------------------------------------------------- calls
    def call(self, tool: str, args: dict, timeout: float = 120) -> dict:
        if not self._session:
            raise MCPError("MCP session not available")
        fut = asyncio.run_coroutine_threadsafe(self._session.call_tool(tool, args), self._loop)
        res = fut.result(timeout)
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"success": not res.is_error, "raw": text}
        self.calls.append({"tool": tool, "args": {k: v for k, v in args.items() if k != "vectors"},
                           "ok": bool(payload.get("success", not res.is_error))})
        if res.is_error or payload.get("success") is False:
            raise MCPError(f"{tool} failed: {payload.get('summary') or payload.get('error') or text[:500]}")
        return payload

    def run_query(self, name: str, params: dict) -> list:
        """Run an installed GSQL query; returns TigerGraph's `results` list."""
        graph = os.getenv("TG_GRAPHNAME", "FraudGraph")
        p = self.call("tigergraph__run_installed_query", {"graph_name": graph, "query_name": name, "params": params})
        return _results(p)


def _results(payload: dict) -> list:
    """tigergraph-mcp wraps REST output in {success, data, ...}; dig out the PRINT list."""
    d = payload.get("data", payload)
    for key in ("results", "result"):
        if isinstance(d, dict) and key in d:
            d = d[key]
    if isinstance(d, dict) and "data" in d:
        d = d["data"]
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except json.JSONDecodeError:
            return []
    return d if isinstance(d, list) else [d]
