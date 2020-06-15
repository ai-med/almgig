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
set -xeu

MODEL_DIR="models/gdb9/almgig/"
DATADIR="data/gdb9/graphs"

python3 train_and_evaluate.py almgig \
	--model_dir "${MODEL_DIR}" \
	--data_dir "${DATADIR}" \
	--dataset "gdb9" \
	--connectivity_penalty_weight 0.005 \
	--discriminator_learning_rate 0.0004 \
	--generator_learning_rate 0.001 \
	--valence_penalty_weight 0.05 \
	--variance_penalty_weight -0.2 \
	--num_latent 96 \
	--epochs 250 \
	--batch_size 512 \
	--beta1 0.5 --beta2 0.9 \
        --reward_type "synthesizability" \
	--temperature 1.0 \
	--weight_gradient_penalty 10.0 \
	--weight_reward_loss 0.0 \
	--weight_reward_loss_schedule "const"

mkdir -p "outputs/descriptors/train"
python -m gan.plotting.compare_descriptors \
	--dist 'emd' \
	--train_file "${DATADIR}/gdb9_train.smiles" \
	--predict_file \
	"models/gdb9/almgig/distribution-learning_model.ckpt-51500.csv" \
	--name "ALMGIG" \
	--palette "stota" \
	-o "outputs/descriptors/train"

mkdir -p "outputs/descriptors/test"
python -m gan.plotting.compare_descriptors \
	--dist 'emd' \
	--train_file "${DATADIR}/gdb9_test.smiles" \
	--predict_file \
	"models/gdb9/almgig/test_distribution-learning_model.ckpt-51500.csv" \
	--name "ALMGIG" \
	--palette "stota" \
	-o "outputs/descriptors/test"

python3 -m gan.plotting.similarity_search \
	--model_dir "${MODEL_DIR}" \
	--data_dir "${DATADIR}" \
	--n_samples 100 \
	-o "outputs/nearest_neighbors/"

python3 -m gan.plotting.interpolate_embedding_grid \
	--model_dir "${MODEL_DIR}" \
	--graph_file "${DATADIR}/gdb9_test.pkl" \
	--n_interpolate 5 \
	-o "outputs/interpolation-test.svg"

python3 -m gan.mol.metrics.get_errors \
	--model_dir "${MODEL_DIR}" \
	--data_dir "${DATADIR}" \
	--latex \
	-o "outputs/errors-table.tex"
