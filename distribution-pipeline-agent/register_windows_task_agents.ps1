# Author: Rui Sun
# Registers a Windows scheduled task for the agent-based distribution forecast pipeline.
# Runs Monday-Friday at the configured time (default 22:30). Weekends are skipped.

param(
    [string]$PythonExe = "python",
    [string]$TaskName = "AgentDistributionForecastNightly",
    [string]$StartTime = "22:30"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainScript = Join-Path $ScriptDir "forecast_with_agents.py"

if (-not (Test-Path $MainScript)) {
    throw "forecast_with_agents.py not found in $ScriptDir"
}

$AgentsDir = Join-Path $ScriptDir "agents"
$ExpectedAgents = @("openai_agents.json", "anthropic_agents.json", "gemini_agents.json")
foreach ($a in $ExpectedAgents) {
    $p = Join-Path $AgentsDir $a
    if (-not (Test-Path $p)) {
        Write-Warning "$a missing in $AgentsDir — run train_agents.py first."
    }
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$MainScript`"" `
    -WorkingDirectory $ScriptDir

$Trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At $StartTime

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Agent-based distribution forecast pipeline (Mon-Fri only, weekends skipped; in-script hash skip avoids AI calls when API data is unchanged)" `
    -Force

Write-Host "Scheduled task registered: $TaskName at $StartTime, Mon-Fri (weekends skipped)"
Write-Host ""
Write-Host "If you want to disable the OLD task (the temperature-draw pipeline):"
Write-Host "  Unregister-ScheduledTask -TaskName 'DistributionForecastNightly' -Confirm:`$false"
