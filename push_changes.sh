#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./push_changes.sh \"your custom commit message\""
  exit 1
fi

COMMIT_MESSAGE="$1"

# Step 1: Push changes in the submodule
echo ">>> Processing submodule: DTensor/Tensor-Implementations"
cd DTensor/Tensor-Implementations || { echo "Error: Could not enter DTensor/Tensor-Implementations"; exit 1; }

# Check for changes in the submodule
if [[ -n $(git status --porcelain) ]]; then
    echo "Staging and committing submodule changes..."
    git add .
    git commit -m "$COMMIT_MESSAGE"
    echo "Pushing submodule changes to branch _adhi_..."
    git push origin _adhi_
else
    echo "No changes detected in submodule."
fi

# Step 2: Push changes in the parent repository
echo ""
echo ">>> Processing parent repository"
cd ../.. || { echo "Error: Could not return to parent directory"; exit 1; }

# Check for changes in the parent repository (including submodule pointer update)
if [[ -n $(git status --porcelain) ]]; then
    echo "Staging and committing parent repository changes..."
    git add .
    git commit -m "$COMMIT_MESSAGE"
    echo "Pushing parent repository changes to branch _adhi_..."
    git push origin _adhi_
else
    echo "No changes detected in parent repository."
fi

echo ""
echo ">>> All done!"
