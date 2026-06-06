#!/bin/bash
set -euo pipefail

SERVICE=${1:-auth-service}
COMMIT_SHA=${2:-unknown}
RELEASE_NAME="${SERVICE}-${COMMIT_SHA}-$(date +%Y%m%d-%H%M%S)"
NAMESPACE="microservices"

echo "=== Rolling out $SERVICE ($RELEASE_NAME) to $ENV ==="

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

case $ENV in
  staging)
    DEPLOYMENT=$SERVICE
    IMAGE_TAG="latest"
    ;;
  production)
    DEPLOYMENT=$SERVICE-canary
    IMAGE_TAG="$COMMIT_SHA"
    ;;
esac

kubectl set image deployment/${DEPLOYMENT} \
  ${SERVICE}=${ECR_REGISTRY}/${SERVICE}:${IMAGE_TAG} \
  -n ${NAMESPACE} \
  --record

echo "Waiting for rollout to complete..."
kubectl rollout status deployment/${DEPLOYMENT} \
  -n ${NAMESPACE} \
  --timeout=600s

echo "=== Rollout complete for $SERVICE ==="

GATEWAY_URL=$(kubectl get svc api-gateway -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "gateway_url=${GATEWAY_URL}" >> "$GITHUB_OUTPUT"
