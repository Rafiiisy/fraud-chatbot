#!/bin/bash

# Docker Stop Script for Fraud Detection Chatbot
# Stops all services and cleans up

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "🛑 Stopping Fraud Detection Chatbot Services"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed."
    exit 1
fi

# Stop services
print_info "Stopping services..."
if docker-compose down; then
    print_status "Services stopped successfully!"
else
    print_error "Failed to stop services"
    exit 1
fi

# Ask if user wants to remove volumes
echo ""
read -p "Do you want to remove persistent data volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Removing volumes..."
    if docker-compose down -v; then
        print_status "Volumes removed successfully!"
        print_warning "All data has been deleted!"
    else
        print_error "Failed to remove volumes"
    fi
else
    print_info "Volumes preserved. Data will be available on next start."
fi

# Ask if user wants to remove images
echo ""
read -p "Do you want to remove Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Removing images..."
    if docker rmi fraud-database:latest fraud-backend:latest fraud-nginx:latest 2>/dev/null; then
        print_status "Images removed successfully!"
    else
        print_warning "Some images may not exist or are in use"
    fi
else
    print_info "Images preserved. They will be reused on next start."
fi

# Show cleanup summary
echo ""
print_info "Cleanup Summary:"
echo "  Services: Stopped"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Volumes: Removed"
    echo "  Images: Removed"
else
    echo "  Volumes: Preserved"
    echo "  Images: Preserved"
fi

echo ""
print_status "Fraud Detection Chatbot services stopped! 🛑"
