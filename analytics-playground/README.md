# Analytics Playground

Data science and analytics project.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Project structure

```
analytics-playground/
├── config/          # Configuration files
├── data/
│   ├── raw/         # Original immutable data
│   ├── processed/   # Cleaned/transformed data
│   └── external/    # External reference data
├── models/          # Trained models
├── notebooks/       # Jupyter notebooks
├── src/             # Source code
│   ├── data.py      # Data loading/saving
│   ├── metrics.py   # Evaluation metrics
│   └── plotting.py  # Visualization utilities
└── tests/           # Unit tests
```
