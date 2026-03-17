#!/bin/bash
set -e
VENV=".venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q -r requirements.txt
streamlit run SENTI.py
