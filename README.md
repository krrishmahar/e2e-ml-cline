# E2E-ML-CLINE

An **end-to-end MLOps / ML workflow demo** that shows how to:

* Train a TensorFlow model (California Housing regression)
* Structure ML code as a proper Python package (`src/` layout)
* Automate code review & improvement with **Cline**
* Run reproducible training inside **Kestra** containers
* Prepare for future observability (metrics, artifacts, orchestration)

This repository is intentionally opinionated toward **real-world ML pipelines**, not notebooks
Checkout out this [JP NOTEBOOK](https://github.com/krrishmahar/jp-notebook) for my notebook resources.

---

## Project Goals

This project exists to demonstrate:

* ✅ Production-style ML project layout
* ✅ Deterministic Python environments (Python **3.11 only**)
* ✅ CI / automation friendliness (Cline, scripts, Kestra)
* ✅ Separation of concerns (data, models, pipelines, utils)
* ✅ Containerized execution without leaking host state

It is **not** meant to be a Kaggle-style experiment repo.

---

## Tech Stack

* **Python**: 3.11 (required)
* **ML**: TensorFlow / Keras
* **Orchestration**: Kestra
* **Automation / Code Review**: Cline, CodeRabbit
* **Environment**: Docker (via Kestra runner)

---

## Repository Structure

```
.
├── src/
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model architectures
│   ├── pipelines/         # Train / run pipelines (entrypoints)
│   ├── utils/             # Callbacks, LR schedules, helpers
│   └── __init__.py
│
├── requirements.txt       # Python dependencies (Py 3.11 compatible)
├── setup.sh               # Local environment bootstrap
├── script.sh              # Automation / Cline workflow helper
├── artifacts/             # Training outputs (created at runtime)
└── README.md
```

All directories under `src/` are **proper Python packages** (via empty `__init__.py`).

---

## Python Version (IMPORTANT)

This project **only supports Python 3.11**.

TensorFlow and several numerical dependencies **do not support Python 3.12+ reliably yet**.

### Check your Python version

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

## Installing Python 3.11

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### macOS (Homebrew)

```bash
brew install python@3.11
```

### Verify

```bash
python3.11 --version
```

---

## Setup (Local Development)

### 1. Clone the repo

```bash
git clone https://github.com/krrishmahar/e2e-ml-cline.git
cd e2e-ml-cline
```

### 2. Run setup script

`setup.sh` creates a virtual environment and installs dependencies.

```bash
chmod +x setup.sh
./setup.sh
```

What `setup.sh` does:

* Creates a Python 3.11 virtual environment
* Activates it
* Installs `requirements.txt`

---

## Running Training Locally

Always run pipelines as **modules**, not files.

```bash
python -m src.pipelines.train
```

This ensures:

* Correct imports (`src.*`)
* Consistent behavior locally and in containers

### Optional arguments

```bash
python -m src.pipelines.train \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --activation relu
```

---

## Automation Scripts

### `script.sh`

Used for **automation workflows**, typically:

* Running Cline prompts
* Running reviews or improvements
* Chaining quality checks

Example:

```bash
chmod +x script.sh
./script.sh
```

This script is intentionally CI-friendly.

---

## Running with Kestra (Recommended)

This project is designed to run **inside Kestra containers**.

### Why Kestra?

* Reproducible execution
* Clear step boundaries
* Artifact handling
* Easy scaling later

---

## Kestra Workflow Design

The workflow is intentionally split into **two steps**:

### Step 1 — Prepare Environment

* Install system deps (git)
* Clone repository
* Install Python dependencies
* Package prepared workspace

This avoids re-installing dependencies during training.

### Step 2 — Train Model

* Restore prepared workspace
* Run training only
* Export artifacts

This keeps training fast and deterministic.

---

## Artifacts

Training produces:

* Saved Keras model (`.h5`)
* Logs
* Metrics

Artifacts are exported by Kestra via:

```yaml
outputFiles:
  - "artifacts/**"
```

---

## Common Pitfalls (Read This)

### ❌ `ModuleNotFoundError: src`

Cause: running Python files directly.

Fix:

```bash
python -m src.pipelines.train
```

---

### ❌ TensorFlow install errors

Cause: wrong Python version.

Fix:

* Use **Python 3.11 only**
* Do not use 3.12 / 3.13 / 3.14

---

### ❌ `git: not found` in Kestra

Cause: `python:slim` images do not include git.

Fix:

```bash
apt-get update && apt-get install -y git
```

---

## Roadmap / Ideas

* GPU training
* Model versioning
* Metrics export (Prometheus / MLflow)
* Slack / Email notifications via Kestra
* CI integration

---

## Author

Made with ❤️ by  **Krrish Mahar**