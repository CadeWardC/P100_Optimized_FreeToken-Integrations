param([switch]$RealEngine, [switch]$Q4Cache, [switch]$Preview, [string]$ModelPath, [int]$TestOutputTokens=-1)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$testData = Join-Path $project ('reports/engine-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testData | Out-Null
function FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()
    return $port
}
$uiPort = FreePort
$enginePort = FreePort
$base = "http://127.0.0.1:$uiPort"
$config = Get-Content "$env:LOCALAPPDATA/FreeTokenDesktop/LlamaCppP100/settings.json" -Raw | ConvertFrom-Json
$model = $config.model
if ($ModelPath) { $model = $ModelPath }
$engine = if ($RealEngine) { $config.server } else { Join-Path $project 'build/desktop/desktop-tests.exe' }
$config.server = $engine
$config.model = ''
$config.port = "$enginePort"
$config.mmproj = ''
$config | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $testData 'settings.json')
function Post($path, $body = @{}) {
    Invoke-RestMethod "$base$path" -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json) -TimeoutSec 15
}
function WaitState($expected) {
    $deadline = (Get-Date).AddSeconds(120)
    do {
        $state = (Invoke-RestMethod "$base/api/state" -TimeoutSec 3).engine
        if ($state.state -eq $expected) { return $state }
        if ($state.state -eq 'error' -and $expected -ne 'error') { throw $state.detail }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $expected"
}
$hostProcess = Start-Process (Join-Path $project 'build/desktop/llamacpp-p100.exe') -ArgumentList @('--no-browser', '--port', "$uiPort", '--data-dir', "`"$testData`"") -WindowStyle Hidden -PassThru
try {
    for ($i = 0; $i -lt 40; $i++) {
        try { $null = Invoke-RestMethod "$base/api/state" -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 200 }
    }
    $rejected = $false
    try { $null = Post '/api/engine/load' } catch {
        if ($_.Exception.Response.StatusCode.value__ -ne 400) { throw }
        $rejected = $true
    }
    if (!$rejected) { throw 'Empty model must reject the load request' }
    'PASS: missing model returns HTTP 400'
    $null = Post '/api/settings' @{model=$model; server=(Join-Path $testData 'missing-server.exe')}
    $null = Post '/api/engine/load'
    $failure = WaitState 'error'
    if ($failure.detail -notmatch 'existing llama-server') { throw 'Missing executable error was lost' }
    'PASS: missing executable reports actionable error'
    $null = Post '/api/settings' @{server=$engine}
    $null = Post '/api/settings' @{max_tokens='-1'; temperature='0.4'; top_p='0.9'; show_output_tokens='1'}
    $null = Post '/api/freetoken/configure' @{action='auto'}
    $autoState = Invoke-RestMethod "$base/api/state"
    if ($autoState.settings.max_tokens -ne '-1') { throw 'Profile overwrote the output limit' }
    $baselineCtx = $autoState.settings.ctx
    $overrideCtx = if ($baselineCtx -eq '2048') { '1024' } else { '2048' }
    $null = Post '/api/settings' @{ctx=$overrideCtx}
    $null = Post '/api/freetoken/configure' @{action='restore'}
    $restored = Invoke-RestMethod "$base/api/state"
    if ($restored.settings.ctx -ne $baselineCtx -or $restored.settings.temperature -ne '0.4' -or $restored.settings.top_p -ne '0.9') { throw 'Profile restore or multi-setting persistence failed' }
    $disk = Get-Content (Join-Path $testData 'settings.json') -Raw | ConvertFrom-Json
    if ($disk.freetoken_baseline.ctx -ne $baselineCtx -or $disk.max_tokens -ne '-1') { throw 'Baseline and no-limit setting were not persisted' }
    'PASS: unlimited output, batch settings, profile override and saved restore'
    if ($Q4Cache) {
        $null = Post '/api/settings' @{cache_k='q4_0';cache_v='q4_0'}
        $kvState = Invoke-RestMethod "$base/api/state"
        if ($kvState.settings.cache_k -ne 'q4_0' -or $kvState.settings.cache_v -ne 'q4_0' -or $kvState.settings.flash -ne 'on') { throw 'Q4 did not configure attention' }
        'PASS: Q4 cache reconfigures memory profile and attention'
    }
    $null = Post '/api/engine/load'
    $null = WaitState 'ready'
    'PASS: retry loads the engine'
    $currentCtx = (Invoke-RestMethod "$base/api/state").settings.ctx
    $overrideCtx = if ($currentCtx -eq '2048') { '1024' } else { '2048' }
    $null = Post '/api/settings' @{ctx=$overrideCtx}
    $null = Post '/api/settings' @{temperature='0.5'}
    if (!(Invoke-RestMethod "$base/api/state").engine.pendingReload) { throw 'Sampling edit cleared pending reload' }
    'PASS: reload requirement survives later inference edits'
    $null = Post '/api/engine/load'
    $null = WaitState 'ready'
    'PASS: reload succeeds while the old engine owns the port'
    if ($TestOutputTokens -gt 0) { $null = Post '/api/settings' @{max_tokens="$TestOutputTokens";reasoning_budget='0'} }
    $null = Post '/api/chat' @{content='Reply with exactly: Hi.'}
    $deadline = (Get-Date).AddSeconds(300)
    do {
        Start-Sleep -Milliseconds 200
        $state = Invoke-RestMethod "$base/api/state"
    } while ($state.engine.generating -and (Get-Date) -lt $deadline)
    if ($state.engine.generating -or !($state.conversation | Where-Object { $_.role -eq 'assistant' -and ($_.content -or $_.reasoning) })) {
        $logs = Invoke-RestMethod "$base/api/logs"
        Write-Output ($logs.lines | Select-Object -Last 5 | ConvertTo-Json -Depth 6)
        throw "Chat did not produce an answer; generating=$($state.engine.generating)"
    }
    'PASS: chat produces an assistant response'
    $reply = $state.conversation | Where-Object role -eq 'assistant' | Select-Object -Last 1
    if ($reply.stats.tokSec -le 0) { throw 'Response speed was not persisted' }
    $originalChat = $state.activeChat
    $originalCount = $state.conversation.Count
    $null = Post '/api/conversation/new'
    $fresh = Invoke-RestMethod "$base/api/state"
    if ($fresh.conversation.Count -ne 0 -or $fresh.chats.Count -lt 2) { throw 'New chat did not preserve history' }
    $null = Post '/api/conversation/select' @{id=$originalChat}
    $restoredChat = Invoke-RestMethod "$base/api/state"
    if ($restoredChat.conversation.Count -ne $originalCount) { throw 'Chat selection lost messages' }
    'PASS: response t/s persists and new chats preserve selectable history' 
    if ($RealEngine -and !($state.conversation | Where-Object { $_.role -eq 'assistant' -and $_.stats.genTokens -gt 0 })) { throw 'Output token count was not saved' }
    $null = Post '/api/engine/unload'
    $null = WaitState 'stopped'
    'PASS: unload stops the engine'
    $other = $state.library.models | Where-Object { $_.id -ne $model } | Select-Object -First 1
    if ($other) {
        $null = Post '/api/settings' @{max_tokens='-1'}
        $null = Post '/api/models/select' @{id=$other.id}
        if ((Invoke-RestMethod "$base/api/state").settings.max_tokens -ne '-1') { throw 'Switching models reset unlimited output' }
        'PASS: model switching preserves output limit preference'
    }
    if ($Preview) { Write-Output "Preview: $base"; $null = Read-Host 'Press Enter when UI verification is complete' }
} finally {
    try { $null = Post '/api/engine/unload' } catch {}
    if (!$hostProcess.HasExited) { Stop-Process -Id $hostProcess.Id }
}
