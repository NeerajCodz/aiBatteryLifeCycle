$ErrorActionPreference = "Stop"

$python = "E:/VIT/aiBatteryLifecycle/venv/Scripts/python.exe"
$token = "MY_TOKEN"
$port = 8888

Write-Host "Starting JupyterLab for MCP on port $port ..."
& $python -m jupyter lab --port $port --IdentityProvider.token $token --ip 0.0.0.0
