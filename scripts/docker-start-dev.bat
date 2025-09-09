@echo off
REM Docker Start Script for Fraud Detection Chatbot (Development)
REM Starts all services using docker-compose.dev.yml

echo 🚀 Starting Fraud Detection Chatbot Services (Development)
echo ==========================================================

REM Change to project root directory
cd /d "%~dp0.."

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ docker-compose is not installed. Please install docker-compose and try again.
    exit /b 1
)

REM Check if dataset directory exists
if not exist "dataset" (
    echo ⚠️  Dataset directory not found. Creating empty dataset directory...
    mkdir dataset\archive
    echo ⚠️  Please add your CSV files to dataset\archive\ directory before starting services
)

REM Check if CSV files exist
if not exist "dataset\archive\fraudTrain.csv" (
    echo ⚠️  CSV files not found in dataset\archive\. Services will start with empty database.
    echo ℹ️  To add data:
    echo ℹ️    1. Place fraudTrain.csv in dataset\archive\
    echo ℹ️    2. Place fraudTest.csv in dataset\archive\
    echo ℹ️    3. Restart services with: docker-compose -f docker-compose.dev.yml restart
)

REM Start services
echo ℹ️  Starting development services with docker-compose...
docker-compose -f docker-compose.dev.yml up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    exit /b 1
)
echo ✅ Services started successfully!

REM Wait for services to be healthy
echo ℹ️  Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo 📊 Service Status:
echo ==================

REM Check database service
docker-compose -f docker-compose.dev.yml ps fraud-database | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo ✅ Database service: Healthy
) else (
    echo ⚠️  Database service: Starting or unhealthy
)

REM Check backend service
docker-compose -f docker-compose.dev.yml ps fraud-backend | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo ✅ Backend service: Healthy
) else (
    echo ⚠️  Backend service: Starting or unhealthy
)

echo.
echo ℹ️  Service URLs:
echo   Backend API: http://localhost:5000
echo   Health Check: http://localhost:5000/health
echo.

echo ℹ️  Useful commands:
echo   View logs: docker-compose -f docker-compose.dev.yml logs -f
echo   Stop services: docker-compose -f docker-compose.dev.yml down
echo   Restart services: docker-compose -f docker-compose.dev.yml restart
echo   View service status: docker-compose -f docker-compose.dev.yml ps
echo.

REM Test the API
echo ℹ️  Testing API endpoint...
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API is responding correctly
) else (
    echo ⚠️  API is not responding yet. It may still be starting up.
    echo ℹ️  Wait a few minutes and try: curl http://localhost:5000/health
)

echo.
echo ✅ Fraud Detection Chatbot (Development) is ready! 🎉
