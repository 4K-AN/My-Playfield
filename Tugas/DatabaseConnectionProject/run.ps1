#!/usr/bin/env pwsh
# PowerShell script untuk compile dan run Java dengan MySQL connector

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

Write-Host "Compiling Java code..." -ForegroundColor Cyan
javac --release 17 -cp "mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar" -d bin src/main.java

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Running application..." -ForegroundColor Cyan
java -cp "bin;mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar" main
