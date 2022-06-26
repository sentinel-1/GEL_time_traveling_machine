#!/bin/bash


VENV_NAME=env_GEL_time_traveling_machine

NOTEBOOK_NAME="GEL Time Machine.ipynb"


SELF=$(python3 -c "import os; print(os.path.realpath('${BASH_SOURCE[0]}'))")
SCRIPT_DIR="$(dirname "${SELF}")"
ENV_BIN="${SCRIPT_DIR}/${VENV_NAME}/bin/"

export JUPYTER_CONFIG_DIR="${SCRIPT_DIR}/.jupyter"


DOCS_DIR="${SCRIPT_DIR}/docs"


##
# Generate HTML
##
"${ENV_BIN}jupyter-nbconvert" "${NOTEBOOK_NAME}" \
  --config "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" \
  --to html --output-dir="${DOCS_DIR}" --output="index" --template OGP_classic

##
# Generate PDF
##
"${ENV_BIN}jupyter-nbconvert" "${NOTEBOOK_NAME}" \
  --embed-images --to pdf --output-dir="${DOCS_DIR}"

