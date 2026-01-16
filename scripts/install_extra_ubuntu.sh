#!/bin/bash
echo "Installing system dependencies for enhanced features..."
sudo apt-get update
sudo apt-get install -y ghostscript python3-tk

# EasyOCR deps usually handled by pip, but libgl1 might be needed for opencv
sudo apt-get install -y libgl1-mesa-glx

echo "Done. Please run: pip install -r requirements-extra.txt"
