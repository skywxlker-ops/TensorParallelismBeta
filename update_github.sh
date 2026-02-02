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
                
                # Attempt to abort any existing rebase to ensure a clean state
                # We use || true to ignore errors if no rebase is in progress
                git rebase --abort >/dev/null 2>&1 || true

                # Detect current branch and remote
                CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
                
                if [ "$CURRENT_BRANCH" == "HEAD" ] || [ -z "$CURRENT_BRANCH" ]; then
                    echo "  Error: Submodule $submodule_path is in detached HEAD state. Skipping."
                    exit 0
                fi

                REMOTE=$(git config branch."$CURRENT_BRANCH".remote || echo "origin")
                
                # Use rebase to keep a clean linear history
                if ! git pull --rebase "$REMOTE" "$CURRENT_BRANCH"; then
                    echo "  Error: Rebase failed in $submodule_path due to conflicts in:"
                    git status --short | grep "^UU"
                    echo "  Please resolve conflicts manually:"
                    echo "  1. Fix files, 2. 'git add <files>', 3. 'git rebase --continue', 4. Re-run this script."
                    exit 1
                fi

                # Stage all changes
                git add .

                # Commit changes
                # simple check if there are changes to commit
                if ! git diff-index --quiet HEAD --; then
                    git commit -m "$COMMIT_MSG"
                    echo "  Changes committed in $submodule_path."
                else
                    echo "  No new changes to commit in $submodule_path."
                fi
                    
                # Push to GitHub
                # Determine current branch
                if [ "$CURRENT_BRANCH" == "HEAD" ] || [ -z "$CURRENT_BRANCH" ]; then
                    echo "  Error: Cannot push while in detached HEAD state in $submodule_path."
                    echo "  Please run 'git checkout <branch_name>' inside the submodule manually."
                else
                    git push origin "$CURRENT_BRANCH" --force-with-lease
                    echo "  Pushed $submodule_path to origin/$CURRENT_BRANCH."
                fi
            )
            echo "Leave submodule: $submodule_path"
            echo "--------------------------------------------------"
        fi
    done
fi

echo "Processing Main Repository..."
echo "--------------------------------------------------"

# Determine current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Pull remote changes and allow unrelated histories (for first-time merges)
git pull origin "$CURRENT_BRANCH" --allow-unrelated-histories || true

# Stage all changes
git add .

# Commit changes
git commit -m "$COMMIT_MSG" || echo "Nothing to commit."

# Push to GitHub
git push origin "$CURRENT_BRANCH"

echo "Update pushed to GitHub successfully!"
