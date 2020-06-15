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
import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def relational_mlp_neighborhood_aggregation(
        inputs,
        num_outputs,
        adjacency,
        layers=2,
        activation_fn=None,
        weights_initializer=tf.glorot_uniform_initializer(),
        biases_initializer=tf.zeros_initializer(),
        scope=None):
    """aggregate features from neighbors"""
    with tf.variable_scope(scope, 'mlp_neigh_agg', [inputs, adjacency]):
        num_edge_types = adjacency.shape[1].value
        edge_layers = []
        for i in range(num_edge_types):
            edge_outputs = slim.repeat(
                inputs, layers, slim.linear,
                num_outputs=num_outputs,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer)
            edge_layers.append(edge_outputs)

        edge_layers = tf.stack(edge_layers, axis=1)
        x = tf.matmul(adjacency, edge_layers)
        x = tf.reduce_sum(x, axis=1)
        if activation_fn is not None:
            x = activation_fn(x)

    return x


@slim.add_arg_scope
def relational_gin_mlp_neighborhood_aggregation(
        inputs,
        num_outputs,
        adjacency,
        layers=2,
        activation_fn=None,
        weights_initializer=tf.glorot_uniform_initializer(),
        biases_initializer=tf.zeros_initializer(),
        scope=None):
    """aggregate features from neighbors"""
    with tf.variable_scope(scope, 'gin_mlp_neigh_agg', [inputs, adjacency]):
        num_edge_types = adjacency.shape[1].value
        edge_layers = []
        for i in range(num_edge_types):
            with tf.variable_scope('relation_{}'.format(i + 1), values=[inputs, adjacency]):
                eps = slim.model_variable('epsilon', shape=[])
                adj_i = adjacency[:, i, :, :]
                x_i = tf.multiply(1. + eps, inputs) + tf.matmul(adj_i, inputs)

                edge_outputs = slim.repeat(
                    x_i, layers, slim.linear,
                    num_outputs=num_outputs,
                    weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer)
                edge_layers.append(edge_outputs)

        edge_layers = tf.stack(edge_layers, axis=1)
        x = tf.reduce_sum(edge_layers, axis=1)
        if activation_fn is not None:
            x = activation_fn(x)

    return x


@slim.add_arg_scope
def graph_aggregation_layer_with_gate(inputs, axis, scope=None):
    with tf.variable_scope(scope, 'graph_aggregation', values=[inputs]):
        gate = slim.fully_connected(inputs,
                                    num_outputs=inputs.shape[-1].value,
                                    activation_fn=tf.nn.sigmoid)
        # apply gate to graph nodes
        x = tf.multiply(inputs, gate)
        # sum over node dimension
        output = tf.reduce_sum(x, axis=axis)

    return output

