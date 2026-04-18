param(
    [string]$PythonExe = "python",
    [string]$TaskName = "DistributionForecastNightly",
    [string]$StartTime = "22:00"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainScript = Join-Path $ScriptDir "forecast_distribution_pipeline.py"

if (-not (Test-Path $MainScript)) {
    throw "forecast_distribution_pipeline.py not found in $ScriptDir"
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$MainScript`" --run-now" `
    -WorkingDirectory $ScriptDir

$Trigger = New-ScheduledTaskTrigger -Daily -At $StartTime

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
    -Description "Nightly multi-model distribution forecast pipeline (100 draws x 3 models x all firms)" `
    -Force

Write-Host "Scheduled task registered: $TaskName at $StartTime daily"
