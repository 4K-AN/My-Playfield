#!/bin/bash
set -euo pipefail
MESSAGE=${1:-"Deployment event"}
COMMIT_SHA=${2:-"unknown"}

WEBHOOK_URL="${SLACK_WEBHOOK_URL}"

curl -X POST "$WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"*Microservices CI/CD*\\nStatus: $GITHUB_JOB\\nEnvironment: $ENV\\nCommit: $COMMIT_SHA\\n$MESSAGE\"}" \
  || echo "Slack notification failed or WEBHOOK_URL not set"
