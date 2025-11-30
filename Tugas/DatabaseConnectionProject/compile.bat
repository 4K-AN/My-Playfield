@echo off
REM Script untuk compile dengan target Java 17
cd /d "%~dp0"
if not exist target\classes mkdir target\classes
javac --release 17 -cp mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar -d target/classes src/main.java
if %ERRORLEVEL% EQU 0 (
    echo Compilation successful with Java 17 target!
) else (
    echo Compilation failed!
    exit /b 1
)

