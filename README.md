# SENTI — SENtence Transformer based data Imputation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Information%20Fusion%202026-orange.svg)](https://doi.org/10.1016/j.inffus.2026.104291)

Zero-shot data imputation for dynamic relational databases using pre-trained sentence transformers.
No model training required.

---

## Requirements

- Python 3.9 or higher
- pip

---

## Dependencies
```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
torch>=2.0.0
transformers>=4.30.0
matplotlib>=3.7.0
plotly>=5.14.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Installation
```bash
# Clone the repository
git clone https://github.com/TariqMahmood93/SENTI-Demo/
cd senti

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure
```
senti/
├── app.py                  # Streamlit entry point
├── impute.py               # CLI entry point
├── requirements.txt
├── senti/
│   ├── __init__.py
│   ├── imputer.py          # Core SENTIImputer class
│   ├── encoder.py          # Sentence transformer wrapper
│   ├── index.py            # FAISS index management
│   ├── evaluator.py        # Metrics (accuracy, F1, RMSE)
│   └── utils.py            # Null injection, preprocessing
├── data/
│   └── examples/           # Sample datasets
├── tests/
│   └── test_imputer.py
└── docs/
    └── architecture.png
```

---

## Citation
```bibtex
@article{ALFANO2026104291,
  title   = {Semantic-Aware Data Imputation in Dynamic Relational Databases
             via Pre-trained Language Models},
  journal = {Information Fusion},
  pages   = {104291},
  year    = {2026},
  issn    = {1566-2535},
  doi     = {https://doi.org/10.1016/j.inffus.2026.104291},
  url     = {https://www.sciencedirect.com/science/article/pii/S1566253526001703},
  author  = {Gianvincenzo Alfano and Sergio Greco and Lucio {La Cava}
             and Tariq Mahmood and Irina Trubitsyna}
}
```
