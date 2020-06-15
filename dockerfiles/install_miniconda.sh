#!/usr/bin/env bash
set -ve

curl -fsSL https://raw.githubusercontent.com/pelson/Obvious-CI/master/bootstrap-obvious-ci-and-miniconda.py > bootstrap-obvious-ci-and-miniconda.py
python3 bootstrap-obvious-ci-and-miniconda.py ~/miniconda x64 3 --without-obvci && source ~/miniconda/bin/activate base

conda config --set show_channel_urls true

conda update -n base -c defaults --yes --quiet conda

conda env create -n tfmol37 --file /tmp/requirements.yaml

. ~/miniconda/bin/activate tfmol37

pip install -q --no-cache-dir --no-deps 'guacamol==0.3.2'

pkg_dir=$(python -c 'import site; print(site.getsitepackages()[0])')
mkdir "${pkg_dir}/fcd"
touch "${pkg_dir}/fcd/__init__.py"

conda install -y -c conda-forge 'scikit-optimize==0.5.2'

mkdir -p ~/.config/matplotlib
echo 'backend: Agg' >> ~/.config/matplotlib/matplotlibrc

conda clean -y --all
rm -fr ~/.cache/pip/
