#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./push_changes.sh \"your custom commit message\""
  exit 1
fi

COMMIT_MESSAGE="$1"

# --- Step 1: Submodule Processing ---
echo ">>> Processing submodule: DTensor/Tensor-Implementations"
cd DTensor/Tensor-Implementations || { echo "Error: Could not enter submodule"; exit 1; }

if [[ -n $(git status --porcelain) ]]; then
    echo "Staging and committing submodule changes..."
    git add .
    git commit -m "$COMMIT_MESSAGE"
    git push origin _adhi_merge_
else
    echo "No changes detected in submodule."
fi

# --- Step 2: Parent Repository Processing ---
echo -e "\n>>> Processing parent repository"
cd ../.. || { echo "Error: Could not return to parent directory"; exit 1; }

# ENSURE LFS TRACKING (Pre-emptive)
echo "Ensuring LFS tracking for large binaries and logs..."
git lfs track "*.nsys-rep" "*.sqlite" "*.bin" "*.pt" "*_exec" \
             "DTensor/TP_MLP_Training_logs/**" \
             "DTensor/gpt2_tp_test/TP_MLP_Torch_Logs/**" > /dev/null
git add .gitattributes
if [[ -n $(git status --porcelain) ]]; then
    echo "Staging and committing parent repository changes..."
    git add .
    git commit -m "$COMMIT_MESSAGE"

    # CRITICAL: Migrate the commit just made to ensure binaries are LFS pointers
    echo "Migrating large files to LFS pointers..."
    git lfs migrate import --include-ref=_adhi_ --include="*.nsys-rep,*.sqlite,*.bin,*.pt,*_exec,DTensor/TP_MLP_Training_logs/**,DTensor/gpt2_tp_test/TP_MLP_Torch_Logs/**" --yes

    echo "Pushing parent repository changes to branch _adhi_..."
    # Use --force because migrate rewrites the local commit history
    git push origin _adhi_ --force
else
    echo "No changes detected in parent repository."
fi

echo -e "\n>>> All done!"
