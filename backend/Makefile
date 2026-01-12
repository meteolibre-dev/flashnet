# FlashNet Backend Makefile
# Common commands for building and running the backend service

.PHONY: help build build-cpu build-debug run run-cpu run-detached logs logs-follow \
        stop clean prune shell test lint typecheck

# Configuration
IMAGE_NAME=flashnet-backend
IMAGE_TAG=cuda
CONTAINER_NAME=flashnet-backend
DOCKERFILE=backend/Dockerfile
COMPOSE_FILE=backend/docker-compose.yml

# Default target
help:
	@echo "FlashNet Backend - Available commands:"
	@echo ""
	@echo "  make build          - Build Docker image with CUDA support (default)"
	@echo "  make build-cpu      - Build Docker image for CPU-only execution"
	@echo "  make build-debug    - Build image with debug mode enabled"
	@echo ""
	@echo "  make run            - Run backend service in foreground"
	@echo "  make run-cpu        - Run CPU version in foreground"
	@echo "  make run-detached   - Run service in background (detached mode)"
	@echo ""
	@echo "  make logs           - Show container logs"
	@echo "  make logs-follow    - Follow logs in real-time"
	@echo ""
	@echo "  make stop           - Stop running container"
	@echo "  make clean          - Remove container (keeps image)"
	@echo "  make prune          - Remove container and image"
	@echo ""
	@echo "  make shell          - Open bash shell in running container"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linter"
	@echo "  make typecheck      - Run type checker"

# Build image with CUDA support
build:
	@echo "Building FlashNet backend with CUDA support..."
	docker build \
		--file $(DOCKERFILE) \
		--target runtime-cuda \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		backend/

# Build CPU-only image
build-cpu:
	@echo "Building FlashNet backend for CPU-only execution..."
	docker build \
		--file $(DOCKERFILE) \
		--target runtime-cpu \
		--tag $(IMAGE_NAME):cpu \
		backend/

# Build with debug mode
build-debug:
	@echo "Building FlashNet backend with debug mode..."
	docker build \
		--file $(DOCKERFILE) \
		--target runtime-cuda \
		--tag $(IMAGE_NAME):debug \
		--build-arg SERVER_DEBUG=true \
		--build-arg SERVER_RELOAD=true \
		backend/

# Run in foreground
run:
	@echo "Starting FlashNet backend..."
	@if [ ! -f backend/.env ]; then \
		echo "Warning: backend/.env not found. Copying .env.example..."; \
		cp backend/.env.example backend/.env; \
	fi
	docker run \
		--name $(CONTAINER_NAME) \
		--rm \
		--gpus all \
		--env-file backend/.env \
		-p 8080:8080 \
		-v $$(pwd)/models:/app/models:ro \
		-v $$(pwd)/backend:/app/backend \
		$(IMAGE_NAME):$(IMAGE_TAG)

# Run CPU version in foreground
run-cpu:
	@echo "Starting FlashNet backend (CPU)..."
	@if [ ! -f backend/.env ]; then \
		echo "Warning: backend/.env not found. Copying .env.example..."; \
		cp backend/.env.example backend/.env; \
	fi
	docker run \
		--name $(CONTAINER_NAME)-cpu \
		--rm \
		--env-file backend/.env \
		-p 8080:8080 \
		-v $$(pwd)/models:/app/models:ro \
		-v $$(pwd)/backend:/app/backend \
		$(IMAGE_NAME):cpu

# Run in detached mode
run-detached:
	@echo "Starting FlashNet backend in detached mode..."
	@if [ ! -f backend/.env ]; then \
		echo "Warning: backend/.env not found. Copying .env.example..."; \
		cp backend/.env.example backend/.env; \
	fi
	docker compose \
		--file $(COMPOSE_FILE) \
		up \
		--detach \
		--build

# Show logs
logs:
	@echo "Showing container logs..."
	docker logs $(CONTAINER_NAME)

# Follow logs in real-time
logs-follow:
	@echo "Following container logs (Ctrl+C to exit)..."
	docker logs -f $(CONTAINER_NAME)

# Stop running container
stop:
	@echo "Stopping FlashNet backend..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"
	@docker compose -f $(COMPOSE_FILE) down 2>/dev/null || true

# Remove container (keeps image)
clean:
	@echo "Cleaning up..."
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || echo "Container already removed"
	@docker rm -f $(CONTAINER_NAME)-cpu 2>/dev/null || echo "CPU container already removed"

# Remove container and image
prune:
	@echo "Pruning all related resources..."
	@make clean
	@echo "Removing image..."
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || echo "Image not found"
	@docker rmi $(IMAGE_NAME):cpu 2>/dev/null || echo "CPU image not found"

# Open bash shell in running container
shell:
	@echo "Opening shell in running container..."
	@docker exec -it $(CONTAINER_NAME) /bin/bash

# Run tests (requires test dependencies)
test:
	@echo "Running tests..."
	@docker run --rm \
		--env-file backend/.env \
		-v $$(pwd):/app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -m pytest backend/tests/ -v

# Run linter
lint:
	@echo "Running linter..."
	@docker run --rm \
		--env-file backend/.env \
		-v $$(pwd):/app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -m ruff check backend/

# Run type checker
typecheck:
	@echo "Running type checker..."
	@docker run --rm \
		--env-file backend/.env \
		-v $$(pwd):/app \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python -m mypy backend/

# Pull latest base images
pull:
	@echo "Pulling latest base images..."
	docker pull python:3.10-slim-bookworm
	@docker pull nvidia/cuda:12.4-runtime-ubuntu22.04 || echo "CUDA image pull skipped"

# Deploy to cloud run (GCP)
deploy-cloud-run:
	@echo "Deploying to GCP Cloud Run..."
	@read -p "Enter region (e.g., us-central1): " REGION; \
	read -p "Enter service name: " SERVICE_NAME; \
	gcloud run deploy $$SERVICE_NAME \
		--image gcr.io/$$(gcloud config get-value project)/$(IMAGE_NAME):$(IMAGE_TAG) \
		--platform managed \
		--region $$REGION \
		--cpu 4 \
		--memory 16Gi \
		--gpu 1 \
		--gpu-type nvidia-tesla-t4 \
		--set-env-vars "SERVER_HOST=0.0.0.0,SERVER_PORT=8080"