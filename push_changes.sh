#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./push_changes.sh \"your custom commit message\""
  exit 1
fi

COMMIT_MESSAGE="$1"

# --- Main Repository Processing ---
echo ">>> Processing parent repository"

git add .
if [[ -n $(git status --porcelain) ]]; then
    echo "Staging and committing repository changes..."
    git commit -m "$COMMIT_MESSAGE"

    echo "Pushing changes to branch _adhi_..."
    git push origin _adhi_
else
    echo "No changes detected to commit."
fi

echo -e "\n>>> All done!"
