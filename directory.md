# MLOps OPPE-2 Practice Directory Structure

Based on the topics covered for OPPE-2:
- Observability, Explainability, Containerizing, Continuous Deployment, Scaling, Monitoring and Security
- CI/CD with GitHub Actions
- Model/Experiment Tracking with MLFlow
- Containerizing with Docker and Artifact Registry
- Deployment Orchestration and scaling with Kubernetes (k8s/GKE)
- MLSecurityOps
- Data and concept drift
- Explainability (SHAP, LIME), Logging, Observability and Performance Monitoring
*(Note: Data Versioning with DVC and Feast Feature Management are excluded)*

```textß
 
 
.
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI Pipeline (Testing, Security Scans, Linting)
│       └── cd.yml                 # CD Pipeline (Docker build/push to Artifact Registry, deploy to GKE)
├── k8s/                           # Kubernetes manifests
│   ├── deployment.yaml            # Deployment for the ML app
│   ├── service.yaml               # Service definition
│   └── hpa.yaml                   # Horizontal Pod Autoscaler for scaling based on CPU/Memory
├── src/                           # Application and ML code
│   ├── api/                       # API for model serving
│   │   ├── app.py                 # FastAPI/Flask application
│   │   └── schemas.py             # Pydantic models for request/response validation
│   ├── data/                      # Data processing and drift detection
│   │   ├── preprocess.py          # Data cleaning and preparation
│   │   └── drift_monitor.py       # Scripts for data and concept drift detection (e.g., using Evidently)
│   ├── models/                    # Model training and prediction
│   │   ├── train.py               # Training script (includes MLflow experiment tracking)
│   │   └── predict.py             # Inference script
│   ├── explainability/            # Model Explainability
│   │   └── explainer.py           # SHAP or LIME integration for explainability
│   └── monitoring/                # Logging and Observability
│       ├── logger.py              # Centralized logging setup
│       └── metrics.py             # Performance monitoring metrics (Prometheus/Grafana integration)
├── tests/                         # Automated testing
│   ├── test_api.py                # Unit and integration tests for API endpoints
│   ├── test_model.py              # Unit tests for ML model predictions
│   └── test_security.py           # MLSecOps specific tests (e.g., testing against data poisoning)
├── Dockerfile                     # Docker configuration for containerization
├── requirements.txt               # Project dependencies
├── data.csv                     # Example raw dataset
├── main.py                        # Entrypoint script
└── README.md                      # Project documentation
```
