# Stroke-IoMT MLOps

A federated machine learning pipeline for stroke and seizure detection using IoMT (Internet of Medical Things) devices.

## Project Structure

```
stroke-iomt-mlops/
├── data/
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed data
├── models/
│   ├── edge/                   # Edge device models (TensorFlow Lite)
│   │   └── tflite_model/
│   ├── fog/                    # Fog computing layer
│   │   └── api_service/
│   └── cloud/                  # Cloud-based federated server
│       └── federated_server/
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── docker/                     # Docker configuration
│   └── Dockerfile
├── .github/workflows/          # CI/CD pipelines
│   └── ml_pipeline.yml
├── requirements.txt
└── README.md
```

## Overview

This project implements a three-tier architecture for distributed ML:
- **Edge**: TensorFlow Lite models on IoMT devices
- **Fog**: Local API service for aggregation
- **Cloud**: Federated learning server

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
```bash
python src/data_preprocessing.py
```

3. Train model:
```bash
python src/train_model.py
```

4. Evaluate model:
```bash
python src/evaluate_model.py
```

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- CUDA 11.8 (optional, for GPU support)

## License

See LICENSE file for details.
