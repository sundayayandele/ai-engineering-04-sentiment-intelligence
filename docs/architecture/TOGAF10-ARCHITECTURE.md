# TOGAF 10-aligned enterprise architecture

This is a **TOGAF-inspired reference**, not a formal TOGAF certification.

## Preliminary and Architecture Vision
Business capability: classify customer sentiment in reviews, support tickets and feedback, with transparent limitations and human review for consequential decisions.

## Business architecture
Actors: support analysts, product managers, ML engineers, privacy officers and platform operators. Value streams: feedback ingestion → review → analysis → escalation → improvement.

## Data architecture
Licensed review datasets; schema/version control; PII minimization; train/validation/test segregation; retention; consent and lineage; curated features and model artifacts.

## Application architecture
FastAPI sentiment API; offline training; optional transformer inference adapter; API gateway, model registry and monitoring as **enterprise reference extensions**, not implemented services.

## Technology architecture
Local Docker/Podman/Rancher labs; GitHub-hosted CI; reference Kubernetes/OpenShift workloads; optional cloud-neutral Terraform/OpenTofu modules.

## Opportunities, migration and governance
Phase 1 CPU baseline → phase 2 pretrained transformer experiment → phase 3 controlled pilot → phase 4 production promotion only after reliability, fairness, privacy and security evidence.

## Architecture decision points
Latency SLO, multilingual needs, false-positive cost, model licensing, GPU vs CPU cost, PII boundaries, deployment sovereignty and data residency.
