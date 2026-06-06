#!/bin/bash
set -euo pipefail

ENV=${1:-production}
PHASE=${2:-canary}
TIMEOUT_SECONDS=${3:-300}
NAMESPACE="microservices"

echo "=== Monitoring $SERVICE in $ENV ($PHASE) for $TIMEOUT_SECONDS seconds ==="

START_TIME=$(date +%s)

while true; do
  ELAPSED=$(($(date +%s) - $START_TIME))
  if [ $ELAPSED -gt $TIMEOUT_SECONDS ]; then
    echo "Monitoring timeout reached after ${ELAPSED}s"
    exit 1
  fi

  DEPLOYMENT=${SERVICE}-canary

  READY_REPLICAS=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' || echo "0")
  DESIRED_REPLICAS=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}' || echo "0")

  ERROR_RATE=$(kubectl get pods -n $NAMESPACE -l app=$SERVICE -o jsonpath='{range .items[*]}{.status.containerStatuses[?(@.state.terminated)]}{"\n"}{end}' 2>/dev/null | grep -c "Error" || echo "0")
  RESTART_COUNT=$(kubectl get pods -n $NAMESPACE -l app=$SERVICE -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' | awk '{sum+=$1} END {print sum+0}')

  SUCCESS_PERCENT=$(kubectl get endpoints $SERVICE -n $NAMESPACE -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -l)

  echo "[$(date +%T)] $SERVICE: ready=$READY_REPLICAS/$DESIRED_REPLICAS, restarts=$RESTART_COUNT, error_pods=$ERROR_RATE"

  if [ "$READY_REPLICAS" -lt "$DESIRED_REPLICAS" ]; then
    echo "Not all replicas ready. Waiting..."
  fi

  if [ "$RESTART_COUNT" -gt 5 ]; then
    echo "High restart count detected: $RESTART_COUNT"
    exit 1
  fi

  SLEEP_INTERVAL=15
  sleep $SLEEP_INTERVAL
done
