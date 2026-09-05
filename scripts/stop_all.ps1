$ErrorActionPreference = 'SilentlyContinue'
# scripts\ 是本脚本所在目录；项目根目录是它的上一级
$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptsDir
$port = 8900

# 1. 停止信号放在项目根目录（Runner 只认根目录的 STOP_SIGNAL）
Set-Content -Path (Join-Path $root 'STOP_SIGNAL') -Value 'STOP' -Encoding ASCII

# 2. 等 Runner 自己退出（它每秒检查一次信号，正常 1~2 秒内退出并写下 phase=stopped）
$pidFile = Join-Path $root 'logs\trading_runner.pid'
$statusFile = Join-Path $root 'logs\runner_status.json'
$deadline = (Get-Date).AddSeconds(5)
while ((Get-Date) -lt $deadline) {
    $stopped = $false
    try {
        $st = Get-Content $statusFile -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($st.phase -eq 'stopped') { $stopped = $true }
    } catch {}
    if ($stopped) { break }
    Start-Sleep -Milliseconds 300
}

# 3. 一次性取全量 python 进程快照（WMI 查询很慢，只查这一次，其余全部在内存完成）
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'")
$byId = @{}
$childrenOf = @{}
foreach ($p in $procs) {
    $id = [int]$p.ProcessId
    $byId[$id] = $p
    $pp = [int]$p.ParentProcessId
    if (-not $childrenOf.ContainsKey($pp)) { $childrenOf[$pp] = @() }
    $childrenOf[$pp] += $id
}

# 4. 认定"本项目进程"：端口持有者及其 python 父进程、Runner pid 文件、
#    命令行里带项目绝对路径的 python；再加上它们在快照里的全部 python 子进程。
$rootPids = New-Object System.Collections.Generic.HashSet[int]
$ownerIds = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($rawId in $ownerIds) {
    $ownerId = [int]$rawId
    [void]$rootPids.Add($ownerId)
    if ($byId.ContainsKey($ownerId)) {
        $parentId = [int]$byId[$ownerId].ParentProcessId
        if ($parentId -gt 0 -and $byId.ContainsKey($parentId)) { [void]$rootPids.Add($parentId) }
    }
}
if (Test-Path $pidFile) {
    $recorded = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($recorded -match '^\d+$') { [void]$rootPids.Add([int]$recorded) }
}
$rootEsc = [regex]::Escape($root)
foreach ($p in $procs) {
    if ($p.CommandLine -and $p.CommandLine -match $rootEsc) { [void]$rootPids.Add([int]$p.ProcessId) }
}

$killSet = New-Object System.Collections.Generic.HashSet[int]
function Add-Tree([int]$startId) {
    if (-not $byId.ContainsKey($startId)) { return }
    if (-not $killSet.Add($startId)) { return }
    foreach ($c in @($childrenOf[$startId])) { Add-Tree $c }
}
foreach ($id in @($rootPids)) { Add-Tree $id }

# 5. 统一结束
foreach ($id in @($killSet)) {
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

Write-Host 'MT5AutoTrader stopped.'
