#!/bin/bash
# push.sh — Simple Git auto-commit and push script

# Exit immediately if a command fails
set -e

# Get the current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Ask for commit message if not provided
if [ -z "$1" ]; then
    read -p "Enter commit message: " COMMIT_MSG
else
    COMMIT_MSG=$1
fi

# Add all changes
git add .

# Commit
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

# Push to the current branch
echo "Pushing to branch '$BRANCH'..."
git push origin "$BRANCH"

echo "✅ Code pushed successfully to branch '$BRANCH'"
