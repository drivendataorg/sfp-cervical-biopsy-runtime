#!/bin/bash
set -e

# GPU or CPU
processor=$1

if [ "$processor" != 'gpu' ] && [ "$processor" != 'cpu' ]; then
    echo "Please pass 'gpu' or 'cpu'. You specified '$1'."
    exit 1
fi

echo "Testing environment py-$processor"
source activate py-$processor
echo "Running Python tests"
python tests/test-installs.py

