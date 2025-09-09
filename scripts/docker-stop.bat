@echo off
REM Docker Stop Script for Fraud Detection Chatbot (Windows)
REM Stops all services and cleans up

echo 🛑 Stopping Fraud Detection Chatbot Services
echo =============================================

REM Change to project root directory
cd /d "%~dp0.."

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ docker-compose is not installed.
    exit /b 1
)

REM Check if docker-compose.dev.yml exists
if not exist "docker-compose.dev.yml" (
    echo ❌ docker-compose.dev.yml not found in current directory.
    echo ℹ️  Make sure you're running this script from the project root.
    exit /b 1
)

REM Stop services
echo ℹ️  Stopping services...
docker-compose -f docker-compose.dev.yml down
if %errorlevel% neq 0 (
    echo ❌ Failed to stop services
    exit /b 1
)
echo ✅ Services stopped successfully!

REM Ask if user wants to remove volumes
echo.
set /p remove_volumes="Do you want to remove persistent data volumes? (y/N): "
if /i "%remove_volumes%"=="y" (
    echo ℹ️  Removing volumes...
    docker-compose -f docker-compose.dev.yml down -v
    if %errorlevel% equ 0 (
        echo ✅ Volumes removed successfully!
        echo ⚠️  All data has been deleted!
    ) else (
        echo ❌ Failed to remove volumes
    )
) else (
    echo ℹ️  Volumes preserved. Data will be available on next start.
)

REM Ask if user wants to remove images
echo.
set /p remove_images="Do you want to remove Docker images? (y/N): "
if /i "%remove_images%"=="y" (
    echo ℹ️  Removing images...
    echo ℹ️  Getting image names from docker-compose...
    for /f "tokens=*" %%i in ('docker-compose -f docker-compose.dev.yml images -q') do (
        echo ℹ️  Removing image: %%i
        docker rmi %%i 2>nul
    )
    echo ✅ Image removal completed!
) else (
    echo ℹ️  Images preserved. They will be reused on next start.
)

REM Show cleanup summary
echo.
echo ℹ️  Cleanup Summary:
echo   Services: Stopped
if /i "%remove_volumes%"=="y" (
    echo   Volumes: Removed
) else (
    echo   Volumes: Preserved
)
if /i "%remove_images%"=="y" (
    echo   Images: Removed
) else (
    echo   Images: Preserved
)

echo.
echo ✅ Fraud Detection Chatbot services stopped! 🛑
