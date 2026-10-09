#!/usr/bin/env bash
# First run: creates a virtual environment and installs everything (a few minutes).
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
streamlit run app.py
