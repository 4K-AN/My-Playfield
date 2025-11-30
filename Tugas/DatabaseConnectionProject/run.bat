@echo off
cd /d "%~dp0"
echo Cleaning old class files...
if exist bin\*.class del /q bin\*.class
if exist target\classes\*.class del /q target\classes\*.class
if not exist target\classes mkdir target\classes
echo Compiling with Java 17 target...
javac --release 17 -cp mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar -d target/classes src/main.java
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    pause
    exit /b 1
)
java -cp target/classes;mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar main
pause
