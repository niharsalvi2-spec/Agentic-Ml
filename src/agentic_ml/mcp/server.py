"""
Model Context Protocol (MCP) Server for Agentic ML Engineering Platform.
"""

import sys
import logging

logger = logging.getLogger("agentic_ml.mcp")

try:
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        # Graceful fallback wrapper if fastmcp package is not installed
        class FastMCP:
            def __init__(self, name: str):
                self.name = name
                self.tools = {}

            def tool(self):
                def decorator(fn):
                    self.tools[fn.__name__] = fn
                    return fn
                return decorator

            def run(self, transport="stdio"):
                print(f"[MCP Server {self.name}] Running on {transport}")

from src.agentic_ml.mcp.tools.run_pipeline import mcp_run_pipeline

mcp = FastMCP("Agentic ML Assistant")

@mcp.tool()
def run_ml_pipeline(task: str) -> str:
    """Run full 10-agent autonomous ML pipeline."""
    return mcp_run_pipeline(task)

if __name__ == "__main__":
    mcp.run(transport="stdio")
