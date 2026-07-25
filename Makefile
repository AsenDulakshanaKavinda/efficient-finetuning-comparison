
# variables
MAIN_PACKAGE_PATH = .
BINARY_NAME = export
BUILD_DIR = ./bin


.PHONY: help
help:
	@echo "Usage: "
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' |  sed -e 's/^/ /'


.PHONY: run_main
## run main.py file
run_main: confirm
	@echo "Running main.py file..."
	uv run main.py


.PHONY: doc_up
## Power up the services
doc_up:
	@echo "Power up the services..."
	docker compose up -d


.PHONY: doc_down
## Power down the services
doc_down: confirm
	@echo "Power down the services..."
	docker compose down

.PHONY: confirm
confirm:
	@echo -n "Are you sure? [y/N] " && read ans && [ "$${ans:-N}" = "y" ]