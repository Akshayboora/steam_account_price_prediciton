.PHONY: install test test-pipeline lint format clean generate-data deploy setup-permissions demo

# Setup
setup-permissions:
	chmod +x scripts/setup_permissions.sh
	./scripts/setup_permissions.sh

install: setup-permissions
	pip install -e ".[dev]"

# Development
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-pipeline:
	pytest tests/test_pipeline.py -v --log-cli-level=INFO

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Data and Model Operations
generate-data:
	python scripts/generate_sample_data.py --n-samples 1000 --category-id 1 --output-dir data/raw/

demo: clean generate-data
	@echo "=== Running complete demo pipeline ==="
	@echo "1. Generating sample data..."
	make generate-data
	@echo "\n2. Training model..."
	python src/train.py --data-path data/raw/sample_train.json --category-id 1 --output-dir models/trained/
	@echo "\n3. Exporting to ONNX..."
	python src/export_onnx.py --model-path models/trained/category_1_model.cbm --output-path models/onnx/category_1_model.onnx
	@echo "\n4. Running evaluation..."
	python scripts/evaluate_model.py --model-path models/onnx/category_1_model.onnx --data-path data/raw/sample_validation.json

deploy:
	python scripts/deploy_model.py --model-path models/onnx/category_1_model.onnx

clean:
	# Clean Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +
	# Clean generated files and directories
	./scripts/cleanup.sh

# Comprehensive cleanup
cleanup-all:
	./scripts/cleanup_all.sh
