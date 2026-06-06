#!/bin/bash
set -euo pipefail

ENV=${1:-production}
PREVIOUS_REVISION=""
NAMESPACE="microservices"

echo "=== Rolling back $SERVICE in $ENV ==="

case $ENV in
  production)
    DEPLOYMENTS=(auth-service-canary order-service-canary api-gateway)
    ;;
  staging)
    DEPLOYMENTS=(auth-service order-service api-gateway)
    ;;
esac

for DEPLOYMENT in "${DEPLOYMENTS[@]}"; do
  echo "Rolling back: $DEPLOYMENT"

  PREVIOUS_REVISION=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE | tail -n 2 | head -n 1 | awk '{print $1}')

  if [ -z "$PREVIOUS_REVISION" ] || [ "$PREVIOUS_REVISION" == "REVISION" ]; then
    echo "No previous revision found for $DEPLOYMENT"
    continue
  fi

  kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE --to-revision=$PREVIOUS_REVISION

  echo "Waiting for $DEPLOYMENT to stabilize..."
  kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=600s

  echo "$DEPLOYMENT rolled back to revision $PREVIOUS_REVISION"
done

echo "=== Rollback complete for $SERVICE in $ENV ==="
