# ArchiMate-inspired viewpoints

Not formal ArchiMate exchange XML; these are **conceptual viewpoints**.

| Layer | Elements | Relationships |
|---|---|---|
| Motivation | Trustworthy sentiment insight, privacy, cost | Drives governance requirements |
| Business | Customer feedback management, support analyst | Business process uses application service |
| Application | Sentiment API, training service, reporting (planned) | API serves classification capability |
| Data | Reviews, labels, model artifact, audit events | Training accesses licensed datasets |
| Technology | Containers, CPU nodes, Kubernetes/OpenShift | Hosts inference service |
| Implementation | GitHub Actions, container build, deployment approval | Realizes architecture increments |

See `diagrams/source/` for editable Mermaid representations.
