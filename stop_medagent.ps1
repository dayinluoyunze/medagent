$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $projectRoot "logs"
$pidFile = Join-Path $logsDir "streamlit.pid"
$portFile = Join-Path $logsDir "streamlit.port"
$targetPids = @()

if (-not (Test-Path $pidFile)) {
    Write-Host "No MedAgent PID file found. Nothing to stop."
    exit 0
}

$pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()

if (-not $pidValue) {
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    Remove-Item $portFile -ErrorAction SilentlyContinue
    Write-Host "PID file was empty and has been removed."
    exit 0
}

$targetPids += [int]$pidValue

$portValue = $null
if (Test-Path $portFile) {
    $portValue = (Get-Content $portFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
}

if ($portValue) {
    try {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $portValue -ErrorAction SilentlyContinue)
        foreach ($listener in $listeners) {
            if ($listener.OwningProcess) {
                $targetPids += [int]$listener.OwningProcess
            }
        }
    } catch {
    }
}

$targetPids = $targetPids | Sort-Object -Unique
$runningProcesses = @()
foreach ($targetPid in $targetPids) {
    $process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if ($process) {
        $runningProcesses += $process
    }
}

if ($runningProcesses.Count -eq 0) {
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    Remove-Item $portFile -ErrorAction SilentlyContinue
    Write-Host "Process was not running. PID file has been removed."
    exit 0
}

foreach ($process in $runningProcesses) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

Remove-Item $pidFile -ErrorAction SilentlyContinue
Remove-Item $portFile -ErrorAction SilentlyContinue

Write-Host ("MedAgent stopped. PIDs: {0}" -f (($runningProcesses | Select-Object -ExpandProperty Id) -join ", "))
