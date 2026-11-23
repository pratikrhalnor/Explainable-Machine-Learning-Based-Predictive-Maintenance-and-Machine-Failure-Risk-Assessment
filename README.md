# Predictive Maintenance ML

## Setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Train
.venv/bin/python -m src.train

## Tests
.venv/bin/python -m pytest -q

## Run app
.venv/bin/streamlit run app/app.py
