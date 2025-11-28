#!/bin/bash
# Automated test runner script

set -e  # Exit on error

echo "========================================="
echo "LangCoach Test Suite"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected. Activating conda env 'lm'...${NC}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate lm || {
        echo -e "${RED}Error: Failed to activate conda environment 'lm'${NC}"
        exit 1
    }
fi

# Install/update test dependencies
echo -e "${GREEN}Installing test dependencies...${NC}"
pip install -q pytest pytest-cov pytest-mock pytest-asyncio coverage

# Run tests with coverage
echo -e "${GREEN}Running tests with coverage...${NC}"
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

# Check coverage threshold
COVERAGE=$(coverage report | grep TOTAL | awk '{print $NF}' | sed 's/%//')
echo -e "${GREEN}Coverage: ${COVERAGE}%${NC}"

if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
    echo -e "${GREEN}✓ Coverage threshold (80%) met!${NC}"
    exit 0
else
    echo -e "${RED}✗ Coverage threshold (80%) not met. Current: ${COVERAGE}%${NC}"
    exit 1
fi

