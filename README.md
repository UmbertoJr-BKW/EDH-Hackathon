# EDIH-Hackathon: The Energy Trilemma Challenge

[🔗 Open in Renku](https://renkulab.io/projects/umberto.mele/smart-design-of-resilient-local-energy-communities-bkw-edh)  

> Can we empower communities to finally break the energy trilemma—maximizing renewables, increase autarky, and guaranteeing network reliability—all at once?

Welcome, innovators! This repository contains the starter code and notebooks for the EDIH Hackathon. Your goal is to design an optimal strategy for reinforcing a local energy grid to prepare it for a sustainable future.

---

## 🎯 The Challenge: Balancing the Energy Trilemma
With access to real datasets from five Swiss low-voltage (LV) networks—including network topologies, household consumption, and simulated PV generation, EV charging, and heat pump load profiles based on Swiss Scenarios for 2050 (15-minute data for one year)—your task is to design, simulate, and rigorously assess novel strategies for local energy systems. We will offer you an evaluator that assesses how good a solution fits for the 3 pillars:
1.  **Minimize Grid Reinforcement Costs**: Keep the financial investment required to upgrade the grid as low as possible.
2.  **Maximize Renewable Energy Installations**: Integrate the maximum amount of renewable energy (like solar PVs) into the grid.
3.  **Maximize Energy Autarky (Self-Sufficiency)**: Reduce the community's reliance on the main power grid by consuming as much locally generated energy as possible.

Your goal is to propose and rigorously assess strategies for achieving an optimal local energy system design. This is an open challenge, so you are free to apply any methodology you find suitable—whether it’s straightforward exploratory analysis or advanced optimization and machine learning techniques—depending on your skills, creativity, and interests. 

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

Renku provides a collaborative, cloud-based environment for running notebooks.

#### How to Start
1.  Click **Login** at the top right corner of [renkulab.io](https://renkulab.io). You can use an edu-ID, GitHub, or ORCID account.
2.  Return to this project's Renku page and click the green **“Launch”** button to start a new session. This may take a few minutes.
3. Paste the sas_url for the data when prompted. Please make sure to **check the box to save credentials**, otherwise, the launching process will **not work**. 
4.  Once inside, you can navigate the notebooks in the file browser on the left.

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