$ErrorActionPreference = "Stop"

$python = "E:/VIT/aiBatteryLifecycle/venv/Scripts/python.exe"
$jupyterUrl = "http://127.0.0.1:8888"
$token = "MY_TOKEN"

Write-Host "Starting jupyter-mcp-server (stdio transport) ..."
& $python -m jupyter_mcp_server --provider jupyter --jupyter-url $jupyterUrl --jupyter-token $token --transport stdio
