#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip
sudo apt-get install -y tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/images
mkdir -p cache/train cache/test
mkdir -p features
mkdir -p checkpoints
mkdir -p results