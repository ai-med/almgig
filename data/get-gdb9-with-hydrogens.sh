#!/bin/bash
# This file is part of Adversarial Learned Molecular Graph Inference and Generation (ALMGIG).
#
# ALMGIG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ALMGIG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ALMGIG. If not, see <https://www.gnu.org/licenses/>.
set -ue

DATA_DIR="gdb9"
RAW_DATA_DIR="${DATA_DIR}/dsgdb9nsd.xyz"
SMI_FILE="${DATA_DIR}/gdb9-smiles.txt"
GRAPHS_DIR="${DATA_DIR}/graphs"
PROJECT_ROOT=".."
PYDATA_DIR="${PROJECT_ROOT}/gan/mol/data"

export PYTHONPATH="${PROJECT_ROOT}"


if [[ ! -d "${RAW_DATA_DIR}" ]]; then
    mkdir -p "${RAW_DATA_DIR}"
    echo -n "Downloading data ..."
    # Download from https://doi.org/10.6084/m9.figshare.978904
    curl -sLL 'https://ndownloader.figshare.com/files/3195389' | tar xjf - -C "${RAW_DATA_DIR}"
    echo " Done."
fi


if [[ ! -f "${SMI_FILE}" ]]; then
    echo -n "Extracting SMILES ..."
    find "${RAW_DATA_DIR}" -name "*.xyz" -type f -print0 | xargs -0 -L 1024 ./xyz2smi.sh > "${SMI_FILE}"
    sort -k2 "${SMI_FILE}" > "${SMI_FILE}.tmp"
    mv "${SMI_FILE}.tmp" "${SMI_FILE}"
    echo " Done."
fi


if [[ ! -d "${GRAPHS_DIR}" ]]; then
    mkdir "${GRAPHS_DIR}"
fi


if [[ ! -f "${GRAPHS_DIR}/gdb9_with_hydrogens_train.pkl" ]]; then
    echo "Creating graphs ..."
    python "${PYDATA_DIR}/smiles2graph.py" \
        --with_hydrogens \
        -i "${SMI_FILE}" \
        -o "${GRAPHS_DIR}" \
        --output_prefix 'gdb9_with_hydrogens'
    echo "Done creating graphs."
fi
