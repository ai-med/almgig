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
import logging
import tensorflow as tf

from nevae.model import Parameters, NeVAEModel
from nevae.io import GraphTransformer, GraphLoader

LOG = logging.getLogger(__name__)


def _get_pkl_input_fn(args):
    from nevae.io import InputFunction

    loader = GraphLoader(args.filename, args.n_nodes)
    graphs = loader.get_all()
    LOG.info("Loaded %d graphs with %d nodes "
             "(max edges %d)", len(graphs),
             loader.num_nodes,
             loader.max_edges)

    transform = GraphTransformer(
        n_nodes=args.n_nodes,
        n_edge_types=3,
        n_node_types=5,
        max_edges=loader.max_edges)
    input_fn = InputFunction(graphs, transform,
                             epochs=args.epochs, n_latent=args.num_latent,
                             shuffle=True)
    return input_fn, loader.max_edges


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('--model_dir', default='ckpt')
    parser.add_argument('--n_nodes', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_latent', type=int, default=5)
    parser.add_argument('--with_masking', action='store_true', default=True)
    parser.add_argument('--max_edges', type=int)
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args=args)

    input_fn, max_edges = _get_pkl_input_fn(args)

    params = Parameters(
        n_nodes=args.n_nodes,
        n_node_types=5,
        n_edge_types=3,
        max_edges=max_edges,
        learning_rate=args.learning_rate,
        num_latent=args.num_latent,
        n_encoder_layers=5,
        weight_decay=2e-4,
        with_masking=args.with_masking,
    )
    model = NeVAEModel(params)

    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=1000,
        save_checkpoints_steps=10000,
        keep_checkpoint_max=2,
    )
    train_hooks = []

    from tensorflow.python import debug as tf_debug

    # train_hooks.append(tf_debug.TensorBoardDebugHook("localhost:8034"))
    # train_hooks.append(tf_debug.LocalCLIDebugHook())

    estimator = tf.estimator.Estimator(model.model_fn, config=config)
    estimator.train(input_fn, hooks=train_hooks)


if __name__ == '__main__':
    main()
