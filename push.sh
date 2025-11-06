#!/bin/bash
set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD)
MSG="${1:-Auto commit on $(date)}"

git add .
git commit -m "$MSG" || echo "No changes to commit."
git pull origin "$BRANCH" --rebase || true
git push origin "$BRANCH"

echo "✅ Code pushed to branch '$BRANCH' without authentication prompt."
