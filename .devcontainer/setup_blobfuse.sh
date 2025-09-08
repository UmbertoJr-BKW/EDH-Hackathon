#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Blobfuse2 setup ---"

# --- Check for the required secret ---
if [ -z "$AZURE_STORAGE_SAS_URL" ]; then
    echo "ERROR: The secret AZURE_STORAGE_SAS_URL is not set."
    echo "Please configure it in the repository's Codespace secrets."
    exit 1
fi

# --- Define Mount Point ---
MOUNT_POINT="//challenge-data/edih-data"

echo "Creating mount point at $MOUNT_POINT"
sudo mkdir -p $MOUNT_POINT
sudo chown vscode:vscode $MOUNT_POINT

# --- Install Microsoft's package repository ---
echo "Installing Microsoft package repository..."
wget https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
sudo apt-get update

# --- Install Blobfuse2 ---
echo "Installing blobfuse2..."
sudo apt-get install -y blobfuse2

# --- Mount the Blob Container ---
echo "Mounting Azure Blob container..."
blobfuse2 mount $MOUNT_POINT --sas-key-env-var=AZURE_STORAGE_SAS_URL

echo "--- Blobfuse2 setup complete! ---"
echo "Your Azure Blob data is now available at: $MOUNT_POINT"
