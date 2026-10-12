# Infrastructure TCO — comparison framework

Illustrative cost drivers only; **no live vendor price quotes**.

| Option | Compute | Fixed overhead | Variable cost | Typical fit |
|---|---|---|---|---|
| Local desktop | Existing CPU/RAM | Hardware and power | Electricity, time | Learning |
| GitHub-hosted runner | Standard runner | Plan/usage rules | Minutes/storage if chargeable | CI, ephemeral |
| Single VPS | 1–2 vCPU CPU VM | Backup and monitoring | Traffic, storage | Small pilot, no HA |
| AWS | EC2/ECS/EKS | Cluster/LB/registry | Egress, GPU, logging | Cloud production |
| Azure | VM/Container Apps/AKS | Gateway, registry | Logs, bandwidth, GPU | Cloud production |
| GCP | Compute/Cloud Run/GKE | Cluster/LB | GPU, network, logs | Cloud production |
| Private cloud | Server amortization | Power, rack, ops staff | Growth and replacement | Data sovereignty |

Monthly TCO = compute + storage + network + orchestration + observability + security + backups + support + staffing + amortized capital. Compare CPU TF-IDF against transformer CPU and GPU options under the same accuracy and latency SLOs. Record region, currency, date, model size and utilization.
