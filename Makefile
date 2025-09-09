# Makefile for Fraud Detection Chatbot Docker Operations

.PHONY: help build start stop restart logs status clean test demo

# Default target
help:
	@echo "Fraud Detection Chatbot Docker Operations"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  start     - Start all services"
	@echo "  stop      - Stop all services"
	@echo "  restart   - Restart all services"
	@echo "  logs      - View logs for all services"
	@echo "  status    - Show service status"
	@echo "  test      - Run tests"
	@echo "  demo      - Run demo"
	@echo "  clean     - Clean up containers and images"
	@echo "  help      - Show this help message"
	@echo ""

# Build all Docker images
build:
	@echo "Building Docker images..."
	docker build -t fraud-database:latest ./database
	docker build -t fraud-backend:latest ./backend
	@echo "Build completed!"

# Start all services
start:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started!"

# Stop all services
stop:
	@echo "Stopping services..."
	docker-compose down
	@echo "Services stopped!"

# Restart all services
restart: stop start

# View logs
logs:
	docker-compose logs -f

# Show service status
status:
	docker-compose ps

# Run tests
test:
	docker-compose exec fraud-backend python test_service.py

# Run demo
demo:
	docker-compose exec fraud-backend python run_demo.py

# Clean up
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker rmi fraud-database:latest fraud-backend:latest fraud-nginx:latest 2>/dev/null || true
	@echo "Cleanup completed!"

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:5000/health || echo "API not responding"
	@docker-compose ps
