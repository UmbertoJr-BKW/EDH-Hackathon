#!/bin/bash

# This is the master script that runs all post-creation steps.
# It is called by devcontainer.json.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- [post-create] Starting setup ---"

# --- 1. Create the Conda environment ---
echo "--- [post-create] Creating Conda environment from environment.yml... ---"
conda env create -f environment.yml
echo "--- [post-create] Conda environment created. ---"

# --- 2. Configure the shell to auto-activate the new environment ---
echo "--- [post-create] Configuring shell for auto-activation... ---"
echo 'conda activate edh-hackathon-env' >> ~/.bashrc
echo "--- [post-create] Shell configured. ---"

# --- 3. Run the data download script ---
echo "--- [post-create] Starting data download script... ---"
# We need to make sure the data script is executable first
chmod +x .devcontainer/setup_data.sh
.devcontainer/setup_data.sh
echo "--- [post-create] Data download script finished. ---"

echo "--- [post-create] All setup steps complete! ---"
