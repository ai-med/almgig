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
import tensorflow.contrib.slim as slim

Scope = Optional[str]


def he_initializer():
    """For ReLU"""
    return tf.variance_scaling_initializer(scale=2.0, distribution='truncated_normal',
                                           mode='fan_in')


def dense(inputs, units, activation=None, **kwargs):
    if 'kernel_initializer' in kwargs:
        kernel_initializer = kwargs.pop('kernel_initializer')
    else:
        if activation is tf.nn.relu:
            kernel_initializer = he_initializer()
        else:
            kernel_initializer = tf.glorot_uniform_initializer()

    if 'bias_initializer' in kwargs:
        bias_initializer = kwargs.pop('bias_initializer')
    else:
        bias_initializer = tf.zeros_initializer()
    use_bias = kwargs.pop('use_bias', True)
    out = tf.layers.dense(inputs, units,
                          activation=None,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          use_bias=use_bias,
                          **kwargs)
    if activation is not None:
        out = activation(out)
    return out


@slim.add_arg_scope
def relational_gin_mlp_neighborhood_aggregation(
        inputs,
        num_outputs,
        adjacency,
        activation_fn=None,
        weights_initializer=tf.glorot_uniform_initializer(),
        biases_initializer=tf.zeros_initializer(),
        use_bias=True,
        scope=None):
    """aggregate features from neighbors"""
    with tf.variable_scope(scope, 'gin_mlp_neigh_agg', [inputs, adjacency]):
        adj_shape = adjacency.get_shape()
        if adj_shape.ndims != 3:
            raise ValueError('Rank mismatch: adjacency (received %s) should have '
                             'rank 3' % (adj_shape.ndims,))
        in_shape = inputs.get_shape()
        if in_shape.ndims != 2:
            raise ValueError('Rank mismatch: inputs (received %s) should have '
                             'rank 2' % (in_shape.ndims,))

        adjacency = tf.cast(adjacency, dtype=inputs.dtype)

        num_edge_types = adjacency.shape[0].value
        edge_layers = []
        for i in range(num_edge_types):
            with tf.variable_scope('relation_{}'.format(i + 1), values=[inputs, adjacency]):
                eps = slim.model_variable('epsilon', shape=[])
                adj_i = adjacency[i]
                x_i = tf.multiply(1. + eps, inputs) + tf.matmul(adj_i, inputs)

                edge_outputs = tf.layers.dense(
                    x_i, num_outputs,
                    activation=None,
                    kernel_initializer=weights_initializer,
                    bias_initializer=biases_initializer,
                    use_bias=use_bias)
                edge_layers.append(edge_outputs)

        edge_layers = tf.stack(edge_layers, axis=1)
        x = tf.reduce_sum(edge_layers, axis=1)
        if activation_fn is not None:
            x = activation_fn(x)

    return x


def mlp_neighborhood_aggregation(
        inputs,
        adjacency,
        activation_fn=None,
        weights_initializer=tf.glorot_uniform_initializer(),
        biases_initializer=tf.zeros_initializer(),
        use_bias=True,
        scope=None):
    in_shape = inputs.get_shape()
    if in_shape.ndims != 2:
        raise ValueError('Rank mismatch: inputs (received %s) must be rank 2/'
                         % (in_shape.ndims,))

    num_outputs = in_shape[1].value
    with tf.variable_scope(scope, 'mlp_neigh_agg', [inputs, adjacency]):
        edge_weights = 1 + tf.argmax(adjacency, axis=0)
        edge_weights = tf.cast(edge_weights, inputs.dtype)
        out_1 = tf.matmul(edge_weights, inputs)

        out_2 = dense(inputs, num_outputs,
                      activation=None, use_bias=use_bias,
                      kernel_initializer=weights_initializer,
                      bias_initializer=biases_initializer)

        outputs = tf.multiply(out_1, out_2)
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs


def kl_div_gaussian_loss(loc, log_variance, scope=None):
    with tf.name_scope(scope, 'KlDivGaussian', [loc, log_variance]):
        value = -0.5 * \
            tf.reduce_mean(tf.reduce_sum(1.0 + log_variance - tf.square(loc)
                                         - tf.exp(log_variance), axis=-1))
    return value


def edge_weights_loss(weight_labels: tf.Tensor,
                      weight_logits: tf.Tensor,
                      add_summaries: bool = True,
                      scope: Scope = None) -> tf.Tensor:
    """Softmax cross-entropy loss of edge weights.

    Parameters
    ==========
    weight_labels : tf.Tensor, shape = (n_adj_lower, n_edge_types)
        One-hot encoding of actually observed edges.
    weight_logits : tf.Tensor, shape = (n_adj_lower, n_edge_types)
        Logits of edge types.
    add_summaries : bool
        Whether to add summary of loss for TensorBoard.
    scope : str
        Name of scope.
    """
    with tf.name_scope(scope, 'nll_edge_weights', [weight_logits, weight_labels]):
        label_shape = weight_labels.get_shape()
        logits_shape = weight_logits.get_shape()
        if logits_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of logits (received %s) should be 2."
                             % (label_shape.ndims,))

        if label_shape.ndims != logits_shape.ndims:
            raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                             "equal rank of logits (received %s)." %
                             (label_shape.ndims, logits_shape.ndims))

        label_shape.assert_is_compatible_with(logits_shape)

        weight_labels = tf.cast(weight_labels, weight_logits.dtype)

        weight_shifted_logits = weight_logits - tf.reduce_max(weight_logits, axis=-1, keepdims=True)
        weight_neg = tf.multiply(1.0 - weight_labels,  weight_shifted_logits)
        sum_exp_weight_neg = tf.reduce_sum(tf.exp(weight_neg), axis=-1, keepdims=True)

        loss_weights = tf.reduce_sum(
            tf.multiply(
                weight_labels,
                tf.log(sum_exp_weight_neg) - weight_shifted_logits),
            axis=-1)
        loss_weights = tf.reduce_mean(loss_weights)
        if add_summaries:
            tf.summary.scalar('loss_edge_weights', loss_weights)
    return loss_weights


def masked_edge_weights_loss(adj_logits: tf.Tensor,
                             weight_labels: tf.Tensor,
                             weight_logits: tf.Tensor,
                             log_weight_indicators: tf.Tensor,
                             true_edge_indices: tf.Tensor,
                             true_edge_list: tf.Tensor,
                             max_edges: int,
                             add_summaries: bool = True,
                             scope: Scope = None) -> tf.Tensor:
    r"""Masked softmax cross-entropy loss of edges.

    For each true edge `(u, v)`, we compute

    .. math::

        loss[u, v] = - \sum_{weight_k} weight_labels[u, v, k] \log p[u, v, k]

        p[u, v, k] = \frac{ edge_mask[u, v]
            exp(adj_logits(node_u, node_v) + weight_logits[node_u, node_v, weight_k])
        }
        { \sum_{node_i, node_j, weight_k}
            exp(adj_logits(node_i, node_j) + weight_logits[node_i, node_j, weight_k])
        }

    Parameters
    ==========
    adj_logits : tf.Tensor, shape = (n_adj_lower, 1)
        Logits of edges.
    weight_labels : tf.Tensor, shape = (n_adj_lower, n_edge_types)
        One-hot encoding of actually observed edges.
    weight_logits : tf.Tensor, shape = (n_adj_lower, n_edge_types)
        Logits of edge types.
    log_weight_indicators : tf.Tensor, shape = (max_edges, n_nodes, n_edge_types)
        `log_weight_indicators[i]` is the logarithm of a binary matrix indicating
        allowed bonds between atoms for the `i`-th bond to be added.
    true_edge_indices : tf.Tensor, shape = (max_edges,)
        Indices of `weight_labels` denoting order of true edges.
    true_edge_list : tf.Tensor, shape = (max_edges, 2)
        true_edge_list[i] contains the indices of nodes forming
        the i-th edge.
    max_edges : int
        Maximum number of edges.
    add_summaries : bool
        Whether to add summary of loss for TensorBoard.
    scope : str
        Name of scope.

    Returns
    =======
    loss : tf.Tensor
        Scalar holding the total loss.
    """
    with tf.name_scope(scope, "nll_edge_weights_masked",
                       [adj_logits, weight_labels, weight_logits, log_weight_indicators,
                        true_edge_indices, true_edge_list]):
        label_shape = weight_labels.get_shape()
        logits_shape = weight_logits.get_shape()

        if label_shape.ndims != logits_shape.ndims:
            raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                             "equal rank of logits (received %s)." %
                             (label_shape.ndims, logits_shape.ndims))

        label_shape.assert_is_compatible_with(logits_shape)
        if adj_logits.get_shape().ndims != 2:
            raise ValueError("Rank mismatch: The rank of adj_logits (received %s) "
                             "should be 2" % (adj_logits.get_shape().ndims,))

        if not adj_logits.get_shape()[0].is_compatible_with(logits_shape[0]):
            raise ValueError("Shapes of weight_logits (received %s) and "
                             "adj_logits (received %s) are incompatible" % (adj_logits.get_shape(), logits_shape))

        ind_shape = log_weight_indicators.get_shape()
        edge_ind_shape = true_edge_indices.get_shape()
        edge_list_shape = true_edge_list.get_shape()

        shapes_fully_defined = (
                ind_shape.is_fully_defined() and
                edge_ind_shape.is_fully_defined() and
                edge_list_shape.is_fully_defined())
        if not shapes_fully_defined:
            raise ValueError("Shape of log_weight_indicators, true_edge_indices and "
                             "true_edge_list must be fully defined")

        if len({ind_shape[0].value, edge_ind_shape[0].value, edge_list_shape[0].value}) != 1:
            raise ValueError("Shape mismatch: The first dimension of label_shape, "
                             "true_edge_indices, and true_edge_list should match")

        if ind_shape.ndims != 3 or ind_shape[2] != label_shape[1]:
            raise ValueError("Shape mismatch: The last dimension of log_weight_indicators (received %s) "
                             "should match the last dimension of label_shape (received %s)" %
                             (ind_shape, label_shape))

        if edge_list_shape.ndims != 2 or edge_list_shape[1] != 2:
            raise ValueError("Shape mismatch: true_edge_list (received %s) should be a 2D tensor "
                             "with shape 2 in the last dimension" % (edge_list_shape,))

        if edge_ind_shape.ndims != 1:
            raise ValueError("Rank mismatch: The rank of true_edge_indices (received %s) "
                             "should be 1" % (edge_ind_shape.ndims,))

        # labels and logits must be of the same type
        weight_labels = tf.cast(weight_labels, weight_logits.dtype)

        logits = weight_logits + adj_logits
        logits_shifted = logits - tf.reduce_max(logits, keepdims=True)
        denominator = tf.log(tf.reduce_sum(tf.exp(logits_shifted), keepdims=True))

        losses = []
        for cur_edge in range(max_edges):
            idx_u, idx_v = tf.unstack(true_edge_list[cur_edge])  # [2]
            mask_u = log_weight_indicators[cur_edge, idx_u]  # [n_edge_types]
            mask_v = log_weight_indicators[cur_edge, idx_v]  # [n_edge_types]
            mask = mask_u + mask_v  # we are in log space, therefore use addition

            # Get label and logits for current edge.
            # If current edge exceeds the actual number of edges,
            # we assume that weight_labels[flat_idx] is all zeros,
            # thus resulting in a zero loss. In practice, `flat_idx = 0`
            # can be used, because the first element of the lower triangular
            # matrix corresponds to a self-loop, which is never observed.
            flat_idx = true_edge_indices[cur_edge]  # scalar
            actual_weight_labels = weight_labels[flat_idx]  # [n_edge_types]

            numerator = logits_shifted[flat_idx] + mask
            loss = tf.reduce_sum(
                actual_weight_labels * (denominator - numerator))
            losses.append(loss)

        total_loss = tf.add_n(losses)
        if add_summaries:
            tf.summary.scalar('loss_edge_weights_masked', total_loss)
    return total_loss
