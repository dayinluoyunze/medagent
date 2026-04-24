$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"
$logsDir = Join-Path $projectRoot "logs"
$stdoutLog = Join-Path $logsDir "streamlit.out.log"
$stderrLog = Join-Path $logsDir "streamlit.err.log"
$pidFile = Join-Path $logsDir "streamlit.pid"
$portFile = Join-Path $logsDir "streamlit.port"
$port = $null

function Test-PortAvailable {
    param([int]$CandidatePort)

    try {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $CandidatePort -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0) {
            return $false
        }
        return $true
    } catch {
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $CandidatePort)
            $listener.Start()
            return $true
        } catch {
            return $false
        } finally {
            if ($listener) {
                $listener.Stop()
            }
        }
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Missing virtualenv python: $pythonExe"
}

if (-not (Test-Path $envFile)) {
    throw "Missing .env file. Configure it before starting MedAgent."
}

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

if (Test-Path $pidFile) {
    $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid) {
        $runningProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($runningProcess) {
            $existingPort = 8501
            if (Test-Path $portFile) {
                $existingPort = (Get-Content $portFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
            }
            Write-Host "MedAgent is already running. PID: $existingPid"
            Write-Host "URL: http://localhost:$existingPort"
            return
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

if (Test-Path $portFile) {
    Remove-Item $portFile -ErrorAction SilentlyContinue
}

foreach ($candidatePort in 8501..8510) {
    if (Test-PortAvailable $candidatePort) {
        $port = $candidatePort
        break
    }
}

if (-not $port) {
    throw "No available port found in range 8501-8510."
}

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.port", $port `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Start-Sleep -Seconds 3

if ($process.HasExited) {
    Write-Host "MedAgent failed to start."
    if (Test-Path $stderrLog) {
        Write-Host "Error log:"
        Get-Content $stderrLog
    }
    exit 1
}

Set-Content -Path $pidFile -Value $process.Id
Set-Content -Path $portFile -Value $port

Write-Host "MedAgent started."
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://localhost:$port"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"

Start-Process "http://localhost:$port" | Out-Null
