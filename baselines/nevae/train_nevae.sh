#!/bin/bash
set -xue

export PYTHONPATH=$(pwd)
mdir="models/nevae-poisson-masked"

python nevae_train_and_generate_sequential.py \
	--model_dir "$mdir" \
	--with_masking \
	--epochs 10
