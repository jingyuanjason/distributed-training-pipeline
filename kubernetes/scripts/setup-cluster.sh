helm install kubeflow-trainer oci://ghcr.io/kubeflow/charts/kubeflow-trainer \
  --namespace kubeflow-system \
  --create-namespace \
  --version 2.1.0 \
  --set runtimes.defaultEnabled=true \
  --wait \
  --timeout 10m

kubectl apply --server-side -k \
  'https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=v2.1.0'

helm repo add kueue https://github.io
helm repo update
helm install kueue kueue/kueue --namespace kueue-system --create-namespace