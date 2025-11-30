@echo off
echo ========================================
echo CLEAN BUILD - Database Connection Project
echo ========================================
echo.

echo [1/3] Cleaning old class files...
if exist bin\*.class del /q bin\*.class
if exist target\classes\*.class del /q target\classes\*.class
echo Done!

echo.
echo [2/3] Compiling with Java 17 target...
javac --release 17 -cp mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar -d bin src/main.java
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)
echo Done!

echo.
echo [3/3] Running application...
echo.
java -cp bin;mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar main

echo.
pause

