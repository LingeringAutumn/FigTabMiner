#!/bin/bash
# FigTabMiner Test Runner
# Run all tests or specific test suites

set -e

echo "============================================"
echo "FigTabMiner Test Suite"
echo "============================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not installed"
    echo "Install with: pip install pytest"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run tests
if [ $# -eq 0 ]; then
    # Run all tests in the tests directory
    echo "Running all tests..."
    pytest "$SCRIPT_DIR" -v
else
    # Run specific test file
    echo "Running $1..."
    pytest "$1" -v
fi

echo ""
echo "============================================"
echo "Tests completed"
echo "============================================"
