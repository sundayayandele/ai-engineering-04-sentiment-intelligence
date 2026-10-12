# Capacity planning and sizing

All values are **starting estimates**, not load-test evidence.

| Environment | CPU | RAM | Free disk | Intended use |
|---|---:|---:|---:|---|
| Baseline API container | 1 vCPU | 1 GiB | 5 GiB | Small CPU TF-IDF inference |
| Learner desktop host | 2–4 logical CPUs | 8 GiB minimum; 16 GiB preferred | 15–25 GiB | Training, desktop VM and API |
| Container VM on Windows/macOS | 2–4 vCPU | 4 GiB minimum; 6–8 GiB preferred | 15–25 GiB | Docker/Rancher/Podman |
| GitHub standard runner | Runner-provided | Runner-provided | Ephemeral | CI and CPU smoke training |
| Pilot reference | 2 replicas × 1 vCPU | 2 GiB total API limits | 20–40 GiB | Functional redundancy, not HA on one node |
| Production reference | 3+ replicas × 1–2 vCPU | 2–4 GiB each | 50+ GiB | Multiple failure domains, benchmark first |
| Transformer experiments | 4+ CPU or suitable GPU | 16+ GiB host RAM | 30+ GiB | Depends strongly on model/token length |

Sizing method: measure p95 latency, RPS, RSS, model cold start, tokenizer cost and model artifact size. Estimate replicas as `ceil(peak_rps / measured_safe_rps_per_replica)`, then add failure-domain and rollout headroom. GPU is optional, not required for the default educational baseline.
