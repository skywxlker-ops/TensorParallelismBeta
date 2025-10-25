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

# Pull remote changes and allow unrelated histories (for first-time merges)
git pull origin main --allow-unrelated-histories || true

# Stage all changes
git add .

# Commit changes
git commit -m "$COMMIT_MSG" || echo "Nothing to commit."

# Push to GitHub
git push origin main

echo "✅ Update pushed to GitHub successfully!"
