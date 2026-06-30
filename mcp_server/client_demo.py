"""
Minimal MCP client for locally verifying mcp_server/server.py - connects
over stdio, discovers tools, and runs each one against your indexed
protocols.

Run after building your index (embeddings/embed_store.py):
    python mcp_server/client_demo.py
"""

import asyncio
import json
from pathlib import Path

from fastmcp import Client

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")


async def verify_and_demo():
    async with Client(SERVER_SCRIPT) as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {(tool.description or '')[:90]}...")

        expected = {"list_protocols", "get_eligibility_criteria", "get_primary_endpoint", "ask_protocol_question"}
        found = {t.name for t in tools}
        missing = expected - found
        if missing:
            print(f"\nWARNING: missing expected tools: {missing}")
        else:
            print("\nAll expected tools present.")

        print("\n--- Demo: list_protocols ---")
        result = await client.call_tool("list_protocols", {})
        print(result.content[0].text if result.content else "(no result)")

        # Pick the first available protocol from the list_protocols result
        # to demo the protocol-specific tools against.
        data = json.loads(result.content[0].text)
        protocols = data.get("protocols") or [{"source_file": f} for f in data.get("files", [])]
        if not protocols:
            print("\nNo protocols found - run ingestion/pdf_extract.py first.")
            return

        sample_protocol = protocols[0]["source_file"]

        print(f"\n--- Demo: get_eligibility_criteria('{sample_protocol}') ---")
        result = await client.call_tool("get_eligibility_criteria", {"protocol_filename": sample_protocol})
        print(result.content[0].text if result.content else "(no result)")

        print(f"\n--- Demo: get_primary_endpoint('{sample_protocol}') ---")
        result = await client.call_tool("get_primary_endpoint", {"protocol_filename": sample_protocol})
        print(result.content[0].text if result.content else "(no result)")

        print("\n--- Demo: ask_protocol_question (no protocol filter, searches all) ---")
        result = await client.call_tool(
            "ask_protocol_question",
            {"question": "Which trial has the largest planned enrollment?"},
        )
        print(result.content[0].text if result.content else "(no result)")


if __name__ == "__main__":
    asyncio.run(verify_and_demo())
