#################################################################################
# GLOBALS                                                                       #
#################################################################################
ENVNAME := .venv
VENV := $(ENVNAME)/bin

PYTHON_INTERPRETER = python

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-35s\033[0m %s\n", $$1, $$2 } /^# [A-Z]/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 3, 70) } ' $(MAKEFILE_LIST)

#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: clean
clean: ## Remove all Python file artifacts and empty directories
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -empty -delete

	rm -rf .*_cache
	rm -rf logs
	rm -rf site

.PHONY: test
test: ## Run tests with pytest
	$(PYTHON_INTERPRETER) -m pytest tests -v -x --dist loadgroup

.PHONY: test_parallel
test_parallel: ## Run tests in parallel with pytest
	$(PYTHON_INTERPRETER) -m pytest tests -n 4 -v --dist loadgroup

.PHONY: lint
lint: ## Lint the codebase using pre-commit hooks
	git add --intent-to-add .
	pre-commit run --all-files

#################################################################################
# DOCS                                                                          #
#################################################################################
.PHONY: serve_docs
serve_docs: ## Serve the documentation locally with MkDocs
	mkdocs serve

.PHONY: build_html_docs
build_html_docs: ## Build the documentation as static HTML
	mkdocs build -d site

#################################################################################
# SETUP                                                                         #
#################################################################################
.PHONY: install
install: install_dependencies ## Install dependencies and pre-commit hooks
	$(MAKE) install_precommit

.PHONY: install_dependencies
install_dependencies: ## Install project dependencies listed in requirements.txt
	@echo "Install dependencies..."
	pip install -r requirements.txt

.PHONY: install_precommit
install_precommit: ## Install pre-commit hooks
	@echo "Install pre-commit hooks..."
	git init -q
	$(VENV)/pre-commit install
	$(VENV)/pre-commit install-hooks
	$(VENV)/pre-commit install --hook-type commit-msg
