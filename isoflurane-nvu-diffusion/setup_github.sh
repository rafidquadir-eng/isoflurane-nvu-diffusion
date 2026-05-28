#!/bin/bash
# =============================================================================
# setup_github.sh
# Run this once to initialise the repo and push to GitHub.
#
# Prerequisites:
#   1. Create an empty repo at https://github.com/new
#      Name: isoflurane-nvu-diffusion
#      Visibility: public
#      Do NOT initialise with README (we already have one)
#
#   2. Set your GitHub username below
#   3. Run:  bash setup_github.sh
# =============================================================================

GITHUB_USER="YOUR_GITHUB_USERNAME"    # ← replace this
REPO_NAME="isoflurane-nvu-diffusion"

cd "$(dirname "$0")"   # run from repo root

git init
git config user.name  "$(git config --global user.name)"
git config user.email "$(git config --global user.email)"

git add .
git commit -m "Initial commit: isoflurane NVU diffusion model v0.2

Multi-compartment cylindrical PDE solver for volatile anesthetic
diffusion from blood vessel to neuron.

Key features:
- Concentration-dependent D_BBB(c) Hill function (v0.2)
- Calibrated and validated vs Tetrault et al. 2008
- 2D multi-vessel visualization
- Sensitivity analysis (4 parameters x 5 values)
- FEniCS FEM scaffold

Key result: K_bt has zero effect on normalized delivery kinetics.
Vessel geometry (R_v) and ECS diffusivity (D_ECS) dominate t90."

git branch -M main
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
git push -u origin main

echo ""
echo "Done! Repo live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
