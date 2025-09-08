#!/bin/bash

# This script is run by the .devcontainer.json's "postCreateCommand".
# It downloads the challenge data from Azure File Storage using azcopy.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- [Data Setup] Starting Azure File Share download ---"

# --- 1. Check for the required secret ---
# === THIS SECTION IS UPDATED TO USE THE NEW SECRET NAME ===
# It now looks for AZURE_FILE_SHARE_SAS_URL.
if [ -z "$AZURE_FILE_SHARE_SAS_URL" ]; then
    echo "ERROR: The secret AZURE_FILE_SHARE_SAS_URL is not set."
    echo "Please configure it in the repository's Codespace secrets."
    exit 1
fi

# --- 2. Define final destination for the data ---
LOCAL_DESTINATION="/workspaces/challenge-data/edih-data/"

# --- 3. Install azcopy (if not already installed) ---
if ! command -v azcopy &> /dev/null
then
    echo "--- [Data Setup] Installing azcopy..."
    wget -q https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
    tar -xvf azcopy.tar.gz
    # The extracted folder has a version number, so use a wildcard
    sudo mv azcopy_linux_amd64_*/azcopy /usr/local/bin/
    rm -rf azcopy.tar.gz azcopy_linux_amd64_*
    echo "--- [Data Setup] azcopy installed successfully."
else
    echo "--- [Data Setup] azcopy is already installed."
fi

# --- 4. Create the local destination directory ---
echo "--- [Data Setup] Creating local destination directory at $LOCAL_DESTINATION"
# We only clean the subdirectory we are about to create, not the whole edih-data folder
rm -rf "${LOCAL_DESTINATION}/data_parquet"
mkdir -p "$LOCAL_DESTINATION"

# --- 5. Format the source URL to copy the subdirectory ---
# The data is in the 'data_parquet' subdirectory.
# We will copy this entire directory to preserve the structure.
SUBDIRECTORY_NAME="data_parquet"
BASE_URL="${AZURE_FILE_SHARE_SAS_URL%%\?*}"
SAS_TOKEN="${AZURE_FILE_SHARE_SAS_URL#*\?}"
SOURCE_URL_TO_SUBDIR="${BASE_URL}/${SUBDIRECTORY_NAME}?${SAS_TOKEN}"

echo "--- [Data Setup] Targeting source directory: ${BASE_URL}/${SUBDIRECTORY_NAME}?..."

# --- 6. Download the data using azcopy ---
echo "--- [Data Setup] Starting download..."
# We copy the subdirectory itself into the LOCAL_DESTINATION.
azcopy copy "$SOURCE_URL_TO_SUBDIR" "$LOCAL_DESTINATION" --recursive=true

echo ""
echo "--- [Data Setup] Download complete! ---"
echo "Data is now available at: ${LOCAL_DESTINATION}${SUBDIRECTORY_NAME}"
