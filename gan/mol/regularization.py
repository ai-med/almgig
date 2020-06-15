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
from typing import Optional
import tensorflow as tf

from .base import MolGANModel


def gradient_penalty(model: MolGANModel,
                     real_adj: tf.Tensor,
                     real_features: tf.Tensor,
                     gen_adj: tf.Tensor,
                     gen_features: tf.Tensor,
                     discriminator_scope: str,
                     gradient_penalty_weight: float = 10.0,
                     target: float = 1.0,
                     add_summaries: bool = False) -> tf.Tensor:
    scope = '{}GradientPenalty'.format(discriminator_scope.replace('/', ''))
    with tf.name_scope(scope,
                       values=[real_adj, real_features, gen_adj, gen_features]):
        batch_size = gen_adj.shape[0].value
        eps = tf.random_uniform(shape=[batch_size])  # batch_size times random numbers

        intp_inputs = {}

        eps_adj = eps
        for _ in range(real_adj.shape.ndims - 1):
            eps_adj = tf.expand_dims(eps_adj, axis=-1)

        intp_inputs['adjacency_in'] = eps_adj * real_adj + (1.0 - eps_adj) * gen_adj

        eps_features = eps
        for _ in range(real_features.shape.ndims - 1):
            eps_features = tf.expand_dims(eps_features, axis=-1)

        intp_inputs['features'] = eps_features * real_features + (1.0 - eps_features) * gen_features

        with tf.name_scope(None):  # Clear scope so update ops are added properly.
            with tf.variable_scope(discriminator_scope, reuse=True):
                disc_interpolates = model.discriminator_fn(intp_inputs, mode=tf.estimator.ModeKeys.TRAIN)

        grads = tf.gradients(disc_interpolates,
                             (intp_inputs['adjacency_in'], intp_inputs['features']))

        penalty = tf.reduce_sum([_penalty_from_gradient(g, batch_size, target=target, scope=sc)
                                 for g, sc in zip(grads, ['adj_grad_pen', 'feat_grad_pen'])])
        gp_weight = tf.convert_to_tensor(gradient_penalty_weight, dtype=penalty.dtype)
        gp_weight = tf.assert_scalar(gp_weight)
        penalty *= gp_weight

        tf.losses.add_loss(penalty)
        if add_summaries:
            tf.summary.scalar('gradient_penalty_loss', penalty)

    return penalty


def _penalty_from_gradient(gradients: tf.Tensor,
                           batch_size: int,
                           epsilon: float = 1e-10,
                           target: float = 1.0,
                           scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'penalty_from_gradient', [gradients]):
        gradient_squares = tf.reduce_sum(
            tf.square(gradients), axis=list(range(1, gradients.shape.ndims)))
        # Propagate shape information, if possible.
        gradient_squares.set_shape([batch_size] + gradient_squares.shape.as_list()[1:])
        # For numerical stability, add epsilon to the sum before taking the square
        # root. Note tf.norm does not add epsilon.
        slopes = tf.sqrt(gradient_squares + epsilon)
        penalties = slopes / target - 1.0

        penalties_squared = tf.square(penalties)
        penalty = tf.reduce_mean(penalties_squared)

    return penalty


def connectivity_penalty(adj: tf.Tensor,
                         features: tf.Tensor,
                         batch_size: int,
                         penalty_weight: float = 1.0,
                         add_summaries: bool = False,
                         scope: Optional[str] = None) -> tf.Tensor:
    def _sigmoid(x, a=100):
        # {1 + exp[−a(x − 1/2 ))] }^−1
        return tf.sigmoid(a * (x - 0.5))

    with tf.name_scope(scope, 'ConnectivityPenalty', [adj, features]):
        n_nodes = adj.shape[-1].value
        with tf.name_scope('adj_power', values=[adj]):
            prob_edge = 1.0 - adj[:, 0, :, :]
            As = [tf.eye(n_nodes, batch_shape=[batch_size]), prob_edge]
            for i in range(2, n_nodes - 1):
                As.append(_sigmoid(tf.matmul(As[i - 1], prob_edge)))
            indicator = _sigmoid(tf.accumulate_n(As))

        prob_node = tf.expand_dims(1.0 - features[:, :, 0], axis=-1)
        # compute all paired probabilities
        q = tf.matmul(prob_node, tf.matrix_transpose(prob_node))
        g = tf.add(q * (1.0 - indicator),
                   (1.0 - q) * indicator)
        penalty = penalty_weight / (n_nodes * n_nodes) * tf.reduce_sum(g)

        tf.losses.add_loss(penalty)
        if add_summaries:
            tf.summary.scalar('penalty', penalty)

    return penalty


def valence_penalty(adj: tf.Tensor,
                    features: tf.Tensor,
                    batch_size: int,
                    valence: tf.Tensor,
                    penalty_weight: float = 1.0,
                    add_summaries: bool = False,
                    scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'ValencePenalty', [adj, features, valence]):
        # valence: max number of allowed edges per node type
        with tf.control_dependencies([tf.assert_rank(valence, 1)]):
            assert valence.shape[0].value == features.shape[-1].value, \
                'dimension mismatch: {} != {}'.format(valence.shape[0].value, features.shape[-1].value)

            n_edge_type = adj.shape[1].value
            n_nodes = adj.shape[2].value

            edge_capacity = tf.reshape(tf.range(n_edge_type, dtype=tf.float32),
                                       [1, n_edge_type, 1, 1])
            diag_mask = tf.expand_dims(1.0 - tf.eye(n_nodes, batch_shape=[batch_size]), axis=1)

            adj_capacity = adj * edge_capacity * diag_mask
            # current number of edges
            num_edges = tf.reduce_sum(adj_capacity, axis=[1, 2])

            # features: [batch_dim, node_idx_dim, node_type_dim]
            max_edges = tf.tensordot(features, valence, [[2], [0]])

            penalty = penalty_weight / n_nodes * tf.reduce_sum(tf.nn.relu(num_edges - max_edges))

            tf.losses.add_loss(penalty)
            if add_summaries:
                tf.summary.scalar('penalty', penalty)

    return penalty


def variance_penalty(x: tf.Tensor,
                     penalty_weight: float = 1.0,
                     add_summaries: bool = False,
                     scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'VariancePenalty', [x]):
        m = tf.reduce_mean(x, axis=0)
        gen_var = penalty_weight * tf.reduce_mean(tf.square(x - m))
        if add_summaries:
            tf.summary.scalar('penalty', gen_var)
    return gen_var
