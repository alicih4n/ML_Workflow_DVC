# Group 3 - ML Workflow with DVC

**Group 3 Submission**

- **Course:** CSCN8010 Foundations of AI
- **Assignment:** ML Workflow with DVC

## Team

- **Ali Cihan Ozdemir (ID: 9091405):** Primary coder
- **Lohith Reddy Danda (ID: 9054470):** Assistant contributor
- **Roshan Bartaula:** No contribution

## Project Overview

This repository contains an end-to-end PyTorch and DVC workflow built around the MNIST dataset. The project uses a three-stage pipeline for dataset preparation, CNN training, and reproducible inference. Model behavior is controlled through `params.yaml`, enabling structured experimentation with activation functions, optimizer choice, momentum, learning rate, batch size, and training epochs.

The implementation is designed for local execution on Apple Silicon systems and automatically uses the Metal Performance Shaders (MPS) backend when it is available. If MPS is unavailable, the code falls back to CUDA or CPU execution.

## Pipeline Stages

1. `prepare`: downloads MNIST and writes processed tensors to `data/processed/`.
2. `train`: trains the CNN, logs first-batch forward/backward instrumentation, writes `model.pt`, and stores evaluation metrics in `metrics.json`.
3. `predict`: loads the trained checkpoint and writes exactly 10 sample predictions to `predictions.json`.

## ML Pipelines and CI/CD

Data Version Control (DVC) is a critical MLOps tool because it extends software versioning principles to data-dependent workflows. In a conventional software CI/CD pipeline, reproducibility is mainly about source code, tests, and deployment artifacts. In machine learning, however, the final result also depends on datasets, parameters, model checkpoints, and execution order across multiple stages. DVC addresses this gap by formalizing pipelines as dependency graphs, tracking which inputs produced which outputs, and rerunning only the stages affected by a change. In a modern CI/CD context, this makes model training and evaluation more auditable, repeatable, and automation-friendly, because every experiment can be tied to explicit code, data, and parameter states rather than undocumented local behavior.

## Project Structure

```text
.
├── data/
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── predict.py
├── params.yaml
├── dvc.yaml
├── requirements.txt
├── README.md
└── MK_Workflow_DVC.ipynb
```

## Running the Workflow

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m pip install -r requirements.txt
/opt/homebrew/Caskroom/miniconda/base/bin/python -m dvc repro
/opt/homebrew/Caskroom/miniconda/base/bin/python -m dvc metrics show
```

## Configurable Experiments

The training stage reads the following parameters from `params.yaml`:

- `epochs`
- `lr`
- `batch_size`
- `activation` with `relu`, `leakyrelu`, or `gelu`
- `optimizer` with `sgd`, `sgd_momentum`, or `adam`
- `momentum` for comparing plain SGD and momentum-based SGD

## Reproducibility Notes

- Large datasets and model artifacts are excluded from Git through `.gitignore`.
- DVC tracks pipeline structure, dependencies, and output metadata in `dvc.yaml` and `dvc.lock`.
- The trained checkpoint stores the activation configuration so inference uses the same network definition as training.
