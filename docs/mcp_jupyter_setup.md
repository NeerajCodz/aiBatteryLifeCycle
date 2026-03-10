# Jupyter MCP Server Setup (Workspace)

This project is configured to use `datalayer/jupyter-mcp-server` inside the local virtual environment.

## Installed packages

- `jupyter-mcp-server==0.22.1`
- `jupyterlab==4.4.1`
- `jupyter-collaboration==4.0.2`
- `jupyter-mcp-tools>=0.1.4`
- `ipykernel`
- `datalayer_pycrdt==0.12.17`

## Workspace config

- MCP config: `.vscode/mcp.json`
- Jupyter start script: `scripts/start_jupyterlab_for_mcp.ps1`
- MCP server start script: `scripts/start_jupyter_mcp_server.ps1`

## Run flow

1. Start JupyterLab:
   - `powershell -ExecutionPolicy Bypass -File scripts/start_jupyterlab_for_mcp.ps1`
2. Start MCP server (if needed manually):
   - `powershell -ExecutionPolicy Bypass -File scripts/start_jupyter_mcp_server.ps1`
3. Ensure token in `.vscode/mcp.json` matches the token used for JupyterLab.

## Notes

- Transport is configured as `stdio`.
- Change URL/token in `.vscode/mcp.json` for your environment.
- For production or remote use, prefer the upstream Docker/HTTP transport guidance.
