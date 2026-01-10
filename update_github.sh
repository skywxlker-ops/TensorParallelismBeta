#!/bin/bash

# ------------------------------
# update_github.sh
# Usage: ./update_github.sh "Your commit message"
# ------------------------------

# Exit immediately if any command fails
set -e

# Check for commit message
if [ -z "$1" ]; then
  echo "Error: Please provide a commit message."
  echo "Usage: ./update_github.sh \"Your commit message\""
  exit 1
fi

COMMIT_MSG="$1"

echo "--------------------------------------------------"
echo "Processing Submodules..."
echo "--------------------------------------------------"

# Iterate through all configured submodules
# We use 'git config' to read .gitmodules directly to ensure we get the paths
if [ -f .gitmodules ]; then
    git config --file .gitmodules --get-regexp path | awk '{ print $2 }' | while read submodule_path; do
        # Skip cgadimpl as we don't have write access
        if [ "$submodule_path" = "cgadimpl" ]; then
            echo "Skipping submodule $submodule_path (third-party dependency)"
            continue
        fi

        if [ -d "$submodule_path" ]; then
            echo "Enter submodule: $submodule_path"
            (
                cd "$submodule_path"
                
                # Pull remote changes to avoid conflicts (optional but recommended)
                # We ignore errors here in case the branch setup is weird or detached head
                git pull origin main --allow-unrelated-histories || echo "Warning: Failed to pull in $submodule_path"

                # Stage all changes
                git add .

                # Commit changes
                # simple check if there are changes to commit
                if git diff-index --quiet HEAD --; then
                    echo "  No changes to commit in $submodule_path."
                else
                    git commit -m "$COMMIT_MSG"
                    echo "  Changes committed in $submodule_path."
                    
                    # Push to GitHub
                    # We assume 'main' is the target branch. 
                    git push origin main
                    echo "  Pushed $submodule_path to origin/main."
                fi
            )
            echo "Leave submodule: $submodule_path"
            echo "--------------------------------------------------"
        fi
    done
fi

echo "Processing Main Repository..."
echo "--------------------------------------------------"

# Pull remote changes and allow unrelated histories (for first-time merges)
git pull origin main --allow-unrelated-histories || true

# Stage all changes
git add .

# Commit changes
git commit -m "$COMMIT_MSG" || echo "Nothing to commit."

# Push to GitHub
git push origin main

echo "Update pushed to GitHub successfully!"
