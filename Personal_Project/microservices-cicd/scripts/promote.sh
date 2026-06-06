#!/bin/bash
set -euo pipefail

ENV=${1:-staging}
NAMESPACE="microservices"

echo "=== Promoting $SERVICE in $ENV to 100% ==="

case $ENV in
  staging)
    echo "Staging auto-promoted during rollout."
    ;;
  production)
    kubectl scale deployment/${SERVICE}-canary --replicas=0 -n ${NAMESPACE} || true
    sleep 5
    kubectl set image deployment/${SERVICE} \
      ${SERVICE}=${ECR_REGISTRY}/${SERVICE}:latest \
      -n ${NAMESPACE}
    kubectl rollout status deployment/${SERVICE} -n ${NAMESPACE} --timeout=600s
    ;;
esac

echo "=== Promotion complete ==="
