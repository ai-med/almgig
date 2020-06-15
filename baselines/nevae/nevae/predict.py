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
import argparse
from pathlib import Path
import pickle
import tensorflow as tf

from nevae.model import NeVAEModel, Parameters
from nevae.io import PredictInputFunction


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='ckpt')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--n_nodes', type=int, required=True)
    parser.add_argument('--num_latent', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1024)
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args=args)

    params = Parameters(
        n_nodes=args.n_nodes,
        n_node_types=5,
        n_edge_types=3,
        max_edges=None,
        learning_rate=None,
        num_latent=args.num_latent,
        n_encoder_layers=5,
        weight_decay=None,
        with_masking=None,
    )
    model = NeVAEModel(params)

    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=None,
        save_checkpoints_secs=100000,
        keep_checkpoint_max=0,
    )

    estimator = tf.estimator.Estimator(model.model_fn, config=config)

    predict_fn = PredictInputFunction(
        n_samples=args.num_samples,
        n_nodes=args.n_nodes,
        n_latent=args.num_latent)
    predictions = []
    for pred in estimator.predict(predict_fn,
                                  yield_single_examples=False):
        predictions.append(pred)

    with args.output.open('wb') as fout:
        pickle.dump(predictions, fout)


if __name__ == '__main__':
    main()
