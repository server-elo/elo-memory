# MCP Setup Guide

Integrate Neuro-Memory-Agent with Claude Code or any MCP-compatible client.

## Quick Start

### 1. Install the Package

```bash
cd neuro-memory-opensource
pip install -e ".[dev]"
```

### 2. Configure MCP Server

Add to your Claude Code MCP settings (usually `~/.config/claude-code/mcp.json`):

```json
{
  "mcpServers": {
    "neuro-memory": {
      "command": "python3",
      "args": ["/path/to/neuro-memory-opensource/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/neuro-memory-opensource/src"
      }
    }
  }
}
```

Or use the installed entry point (if you add one to pyproject.toml):

```json
{
  "mcpServers": {
    "neuro-memory": {
      "command": "neuro-memory",
      "args": ["mcp"]
    }
  }
}
```

### 3. Available Tools

Once connected, the MCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `store_memory` | Store a new memory episode with auto-generated embeddings |
| `retrieve_memories` | Retrieve relevant memories by query text or embedding |
| `consolidate_memories` | Run memory consolidation (sleep-like replay) |
| `get_stats` | Get memory system statistics |

### 4. Example Usage

In Claude Code:

```
Store this memory: "User prefers dark mode interfaces"
```

```
Retrieve memories about: "user interface preferences"
```

```
Run memory consolidation
```

## Manual Testing

Test the MCP server directly:

```bash
echo '{"method": "get_stats", "params": {}}' | python3 mcp_server.py
```

## Troubleshooting

**Import errors?** Make sure you installed with `pip install -e ".[dev]"`

**Module not found?** Check that `src/` is in your PYTHONPATH

**Embeddings not working?** Install sentence-transformers: `pip install sentence-transformers`