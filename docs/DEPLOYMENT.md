# Deployment paths

Local: train host smoke model then mount artifacts read-only into Docker Compose or Podman.

GitHub Actions: quality, synthetic training, FastAPI live prediction and container build; not persistent hosting.

Kubernetes/OpenShift: replace image placeholder, prepopulate approved model PVC, configure multi-node storage, TLS gateway/auth, restricted security context, metrics and rollout policy. Starter manifests are not a validated HA deployment.