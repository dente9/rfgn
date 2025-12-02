# ================= 配置区域 =================
# 参数：分钟数
$minutesLater = 180  # 设置为180分钟（3小时）

# 定义任务名称
$taskName = "ShutDownTask"

# 定义要执行的 API 命令
$curlCommand = 'curl -X POST https://xn-a.suanjiayun.com:4333/container/api/projects/6893007cd7155fa16b8b4e70/instances/691ed618268def75ee640f5c/6893007cd7155fa16b8b4e72/shutDown  -H "Authorization: nR7dcDhUyp1IPcHrWrnxDM627o8E4ROE3FbyeC72jaICH4LEqFeIy5QgdJzQPMO3" -H "Content-Type: application/json" '
# ===========================================

# 1. 计算时间
$currentDateTime = Get-Date
$targetDateTime = $currentDateTime.AddMinutes($minutesLater)
$targetTime = $targetDateTime.ToString("HH:mm")
$targetDate = $targetDateTime.ToString("yyyy/MM/dd")

# 2. 准备批处理文件 (关键修改：把删除任务的命令写在 bat 里，而不是靠 schtasks /Z)
$batchFilePath = "$env:TEMP\shutdown_task.bat"
# bat 内容：执行 curl，然后执行删除任务命令
$batchContent = "@echo off`n$curlCommand`nschtasks /delete /tn `"$taskName`" /f"
Set-Content -Path $batchFilePath -Value $batchContent

# 3. 检查任务是否存在
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

# 定义创建任务的函数 (去掉了导致报错的 /Z 参数)
function Create-MyTask {
    Write-Host "正在创建新任务..." -ForegroundColor Cyan
    # /SC ONCE: 执行一次 /ST: 时间 /SD: 日期 /F: 强制创建
    schtasks /create /tn $taskName /tr $batchFilePath /sc once /st $targetTime /sd $targetDate /F | Out-Null
    Write-Host "计划任务已创建！将在 $targetDate $targetTime 执行操作。" -ForegroundColor Green
    Write-Host "注意：任务执行成功后，批处理脚本将自动删除该任务。" -ForegroundColor Gray
}

# 4. 核心逻辑分支
if ($null -eq $existingTask) {
    # --- 情景 1: 任务不存在 ---
    Write-Host "任务 $taskName 不存在。"
    Create-MyTask
}
else {
    # --- 情景 2: 任务已存在 ---
    # 修复：增加判空处理，防止报错
    if ($null -ne $existingTask.NextRunTime) {
        $nextRun = $existingTask.NextRunTime.ToString("yyyy/MM/dd HH:mm:ss")
    }
    else {
        $nextRun = "未知 (可能已过期或权限不足)"
    }

    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "警告：任务 $taskName 已存在！" -ForegroundColor Yellow
    Write-Host "当前计划执行时间: $nextRun" -ForegroundColor Yellow
    Write-Host "----------------------------------------"
    Write-Host "按 'r' 键删除任务，或等待 3秒 后覆盖并更新任务时间..." -NoNewline

    $timeout = 3 # 等待秒数
    $startTime = Get-Date
    $userPressedR = $false

    # 循环检测按键
    while ((Get-Date) -lt $startTime.AddSeconds($timeout)) {
        if ([System.Console]::KeyAvailable) {
            $key = [System.Console]::ReadKey($true).Key
            if ($key -eq "R") {
                $userPressedR = $true
                break
            }
        }
        Start-Sleep -Milliseconds 100
    }
    Write-Host "" # 换行

    if ($userPressedR) {
        # 用户按了 R，只删除
        Write-Host "检测到输入 'r'，正在删除任务..." -ForegroundColor Red
        schtasks /delete /tn $taskName /f | Out-Null
        Write-Host "任务已删除。" -ForegroundColor Red
    }
    else {
        # 超时，删除旧的并创建新的
        Write-Host "等待超时，正在更新任务..." -ForegroundColor Cyan
        schtasks /delete /tn $taskName /f | Out-Null
        Create-MyTask
    }
}