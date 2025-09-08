# EDIH-Hackathon: The Energy Trilemma Challenge

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/{{ YOUR_GITHUB_ORG }}/{{ YOUR_REPO_NAME }})

> Can we empower communities to finally break the energy trilemma—maximizing renewables, increase autarky, and guaranteeing network reliability—all at once?

Welcome, innovators! This repository contains the starter code and notebooks for the EDIH Hackathon. Your goal is to design an optimal strategy for reinforcing a local energy grid to prepare it for a sustainable future.

---

## 🎯 The Challenge: Balancing the Energy Trilemma

Your mission is to develop a solution that finds the best possible balance between three competing objectives for a local energy community:

1.  **Minimize Grid Reinforcement Costs**: Keep the financial investment required to upgrade the grid as low as possible.
2.  **Maximize Renewable Energy Installations**: Integrate the maximum amount of renewable energy (like solar PVs) into the grid.
3.  **Maximize Energy Autarky (Self-Sufficiency)**: Reduce the community's reliance on the main power grid by consuming as much locally generated energy as possible.

You will use the provided notebooks and functions to simulate different scenarios, evaluate their performance, and find a set of optimal solutions that represent the best trade-offs.

## 📂 Repository Structure

This repository is organized to guide you through the challenge step-by-step.

```
EDH-Hackathon/
├── .devcontainer/                # Configuration for GitHub Codespaces
├── environment.yml                 # Conda environment definition
├── src/
│   ├── functions.py                # Core logic for grid simulation and evaluation
│   └── visualization.py            # Plotting and visualization helpers
├── plots/                          # A place to save your generated plots
├── 1_LocalGrids.ipynb              # Explore the grid topology
├── 2_ProfilesExploration.ipynb     # Understand consumption and generation profiles
├── 3_GridReinforcement.ipynb       # See how the grid reinforcement algorithm works
├── 4_BatteriesAndCurtailment.ipynb # Learn to use storage and curtailment
├── 5_TrilemmaEvaluation.ipynb      # The core notebook for evaluating your solutions
├── 6_P2H2.ipynb                    # (Advanced) Use Power-to-Hydrogen for autarky
└── 7_PVsMaxCapacity.ipynb          # Physical limits for PV installations
```

---

## 🚀 Getting Started: Setting Up Your Environment

We support three ways to set up your environment. Choose the one that works best for you.

### Option A: GitHub Codespaces (Recommended)

This is the easiest and fastest way to get started. It creates a complete, pre-configured development environment in your browser in minutes, with the data already mounted.

1.  Click the **"Open in GitHub Codespaces"** badge at the top of this README.
2.  Alternatively, click the green **`< > Code`** button on the main repository page.
3.  Go to the **Codespaces** tab and click **"Create codespace on main"**.

The environment will build automatically. When it's ready, your Conda environment will be active, and the challenge data will be available in the `/workspaces/azure_data` directory.

### Option B: Renku (Cloud Notebook Platform)

Renku provides a collaborative, cloud-based environment for running notebooks.

#### How to Start
1.  Click **Login** at the top right corner of [renkulab.io](https://renkulab.io). You can use an edu-ID, GitHub, or ORCID account.
2.  Return to this project's Renku page and click the green **“Launch”** button to start a new session. This may take a few minutes.
3.  Once inside, you can navigate the notebooks in the file browser on the left.

#### Managing Your Session
-   **Pause Session**: Click the Pause button (top-left) to save your workspace state for up to two weeks. Your session will also auto-pause after 2 hours of inactivity.
-   **Shut Down Session**: This will terminate the session. **Only files saved in a Data Connector or pushed to your Git repository will be kept.** All other changes will be lost.
-   **Download Your Work**: To be safe, always download important notebooks and data to your local machine by right-clicking them in the file explorer.

#### Updating Your Repository
If the main hackathon repository is updated, you can pull the latest changes into your running session.
1.  Go to the **Git** tab in the left menu bar.
2.  Click on **"Pull latest changes"**.

#### Frequently Asked Questions
-   **Q: "Can I use Renku without logging in?"**  
    Yes, but you will not be able to pause your session.
-   **Q: "I do not have a 'Pause Session' option."**  
    Please ensure you are logged in. Only logged-in users can pause sessions.

### Option C: Local Development (Advanced)

This method gives you full control but requires you to install everything on your own machine.

#### Step 1: Install Miniconda
First, you need a Conda installation. We recommend Miniconda.

```bash
# Download the installer for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer script
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the on-screen prompts. We recommend accepting the default settings, including the final prompt to run `conda init`. This will configure your shell to use Conda.

**IMPORTANT**: After the installation finishes, you **must close and reopen your terminal** for the changes to take effect.

#### Step 2: Clone the Repository
```bash
git clone https://github.com/{{ YOUR_GITHUB_ORG }}/{{ YOUR_REPO_NAME }}.git
cd EDH-Hackathon
```

#### Step 3: Create and Activate the Conda Environment
In a new terminal, navigate to the repository folder and run the following commands.

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the newly created environment
conda activate edh-hackathon-env
```

#### Step 4: Link the Environment to Jupyter
To make sure Jupyter Notebook (and VS Code) can find your new environment, install a kernel spec.

```bash
# Install the ipykernel package in your new environment
conda install -c conda-forge ipykernel

# Create a kernel spec for Jupyter
python -m ipykernel install --user --name edh-hackathon-env --display-name "EDH Hackathon (Conda)"
```
You can now open the notebooks in Jupyter or VS Code and select the "EDH Hackathon (Conda)" kernel.

---

## 💾 Data Setup

To keep this repository lightweight, the challenge data is provided separately and **should not be committed to Git**.

1.  **Download the data** from the provided link: `[Link to your Azure Blob Storage or other data source]`
2.  Unzip the data if necessary.
3.  Organize your folders so that the data directory (`challenge-data`) and the repository directory (`EDH-Hackathon`) are at the same level (i.e., they are siblings).

Your final directory structure should look like this:
```
your_workspace/
├── challenge-data/
│   └── edih-data/
│       ├── data_file_1.parquet
│       └── ...
└── EDH-Hackathon/      <-- This is where you cloned the repo
    ├── src/
    ├── 1_LocalGrids.ipynb
    └── ...
```

**Note for Codespaces Users**: This step is done for you automatically! The data will be mounted and available at `/workspaces/azure_data/edih-data/`.

---

## 🏆 Submission

All final submissions must be made through our official hackathon platform.

**Live Platform:** [https://legs-challenge-bkw.onrender.com](https://legs-challenge-bkw.onrender.com)

The platform is designed to provide a seamless experience, allowing you to focus on your solution while receiving instant feedback.

#### Key Features
-   **User Authentication**: Sign up and log in to manage your submissions securely.
-   **File Submission**: Upload your solution, consisting of 9 `.parquet` files (7 required, 2 optional).
-   **Automated Evaluation**: Your submission is automatically processed to calculate the three objective scores (Cost, Renewables, Autarky).
-   **Interactive Leaderboard**: View all results on a live leaderboard that can be sorted by any of the three objectives.
-   **3D Score Visualization**: Explore the entire solution space in an interactive 3D plot, which highlights the Pareto optimal frontier.