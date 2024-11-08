@echo off
echo Activating virtual environment...

REM Activate the virtual environment
call venv\Scripts\activate

REM Check if activation was successful
if "%VIRTUAL_ENV%"=="" (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Virtual environment activated.

REM Run the tests
echo Running tests...
python src/run_tests.py

REM Check if tests ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Tests failed.
    exit /b 1
)

echo Tests completed successfully.

REM Deactivate the virtual environment
deactivate
echo Virtual environment deactivated.

REM Optional: Show location of test results
echo Test results are available in the data\results directory.
