@echo off
REM Docker Build Script for Fraud Detection Chatbot (Development)
REM Builds minimal Docker images for faster development

echo 🐳 Building Fraud Detection Chatbot Docker Images (Development)
echo ================================================================

REM Change to project root directory
cd /d "%~dp0.."

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Check if dataset directory exists
if not exist "dataset" (
    echo ⚠️  Dataset directory not found. Creating empty dataset directory...
    mkdir dataset\archive
    echo Please add your CSV files to dataset\archive\ directory
)

REM Build database image
echo Building database image...
docker build -t fraud-database:latest ./database
if %errorlevel% neq 0 (
    echo ❌ Failed to build database image
    exit /b 1
)
echo ✅ Database image built successfully

REM Build backend image (development)
echo Building backend image (development)...
docker build -t fraud-backend:dev -f ./backend/Dockerfile.dev ./backend
if %errorlevel% neq 0 (
    echo ❌ Failed to build backend image
    exit /b 1
)
echo ✅ Backend image built successfully

echo.
echo ✅ All Docker images built successfully!
echo.
echo Available images:
docker images | findstr fraud

echo.
echo To start the development services, run:
echo   docker-compose -f docker-compose.dev.yml up -d
echo.
echo To view logs, run:
echo   docker-compose -f docker-compose.dev.yml logs -f
echo.
echo To stop the services, run:
echo   docker-compose -f docker-compose.dev.yml down
