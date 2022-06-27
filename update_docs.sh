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

##
# Update custom 404 page
##
UPDATED_404_TODAY=$(python3 <<EOF
import os
from datetime import datetime as dt
from datetime import timedelta as td
print((dt.now() - dt.fromtimestamp(os.path.getctime("docs/404.html"))) < td(days=1))
EOF
)
# Rate limiting to one download per day:
if [ "${UPDATED_404_TODAY}" != "True" ];then
    echo "Updating custom 404 page"
    rm -f "${DOCS_DIR}/404.html"
    wget --output-document="${DOCS_DIR}/404.html" "https://raw.githubusercontent.com/sentinel-1/sentinel-1.github.io/master/docs/404.html"
else
    echo "SKIPPING custom 404 page update"
fi
