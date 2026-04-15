.PHONY: test lint format build clean

test:
	pytest --cov=lossless_claw --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

build:
	hatch build

clean:
	rm -rf dist build *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
