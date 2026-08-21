"""
MCP Server Entrypoint for Agentic ML Engineering Platform.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.agentic_ml.mcp.server import mcp

if __name__ == "__main__":
    print("Starting Model Context Protocol (MCP) Server for Agentic ML Platform on stdio...")
    mcp.run(transport="stdio")
