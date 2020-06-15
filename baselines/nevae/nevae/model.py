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
from collections import namedtuple
import logging
from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import nevae.layers as layers
from nevae.io import GraphData

LOG = logging.getLogger(__name__)


class Parameters(namedtuple('Parameters',
                            ('n_nodes',
                             'n_node_types',
                             'n_edge_types',
                             'max_edges',
                             'learning_rate',
                             'num_latent',
                             'n_encoder_layers',
                             'weight_decay',
                             'with_masking'))):
    pass


EncoderOutputs = namedtuple('EncoderOutputs',
                            ('latent_vec', 'mu', 'log_variance'))

DecoderOutputs = namedtuple('DecoderOutputs',
                            ('edge_logits',
                             'edge_weight_logits',
                             'feature_logits',
                             'num_edges_log_rate',))


def unravel_index(idx, shape):
    """TensorFlow equivalent to np.unravel_index

    See `unravel_index_loop_corder` in
    https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/compiled_base.c
    """
    ret = []
    val = idx
    for dim in reversed(shape):
        ret.append(tf.floormod(val, dim))
        val = tf.floor_div(val, dim)
    return ret[::-1]


class NeVAEModel:

    def __init__(self, params: Parameters) -> None:
        self._params = params
        self._is_training = None

    def model_fn(self,
                 features: Dict[str, tf.Tensor],
                 labels: Dict[str, tf.Tensor],
                 mode: str) -> tf.estimator.EstimatorSpec:
        self._is_training = mode == tf.estimator.ModeKeys.TRAIN

        if mode == tf.estimator.ModeKeys.PREDICT:
            kwargs = self.predict_model_fn(features)
        else:
            kwargs = self.train_model_fn(features, labels)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            **kwargs)

    def predict_model_fn(self,
                         features: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        latent_vec = features['noise']

        decoder = Decoder(
            n_nodes=self._params.n_nodes,
            n_features=self._params.n_node_types,
            n_edge_types=self._params.n_edge_types,
            num_latent=self._params.num_latent,
            add_summaries=True
        )

        with tf.variable_scope('Decoder'):
            dec_out = decoder.call(latent_vec, self._is_training)

        dist = self._get_num_edge_distribution(
            log_rate=dec_out.num_edges_log_rate)
        num_edge_pred = dist.mean()

        with tf.name_scope('Prediction'):
            prob_features = tf.nn.softmax(dec_out.feature_logits, axis=1)
            # prob_edge_weight = tf.nn.softmax(dec_out.edge_weight_logits, axis=1)
            # shifted_logits = dec_out.edge_logits - tf.reduce_max(dec_out.edge_logits, keepdims=True)
            # edge_exp = tf.exp(shifted_logits)
            # prob_edge = edge_exp / tf.reduce_sum(edge_exp)

        predictions = {
            'logits_edge_weight': dec_out.edge_weight_logits,
            'logits_edge': dec_out.edge_logits,
            'prob_features': prob_features,
            'log_rate_num_edges': dec_out.num_edges_log_rate,
            'num_edges': num_edge_pred,
        }

        return {'predictions': predictions}

    def train_model_fn(self,
                       features: Dict[str, tf.Tensor],
                       labels: Dict[str, tf.Tensor]):
        data = GraphData(
            adjacency=tf.identity(features['adjacency'], name='adjacency'),
            adjacency_lower=tf.identity(features['adjacency_lower'], name='edges'),
            edge_weights=tf.identity(features['edge_weights'], name='edge_weights'),
            features=tf.to_float(features['features'], name='features'),
            log_weight_indicator=features['log_weight_indicator'],
            edge_index=features['edge_index'],
            edge_list=features['edge_list'],
            num_edges=features['num_edges'])
        noise = labels

        encoder = NeVaeEncoder(
            num_latent=self._params.num_latent,
            n_layers=self._params.n_encoder_layers,
            add_summaries=True
        )

        with tf.variable_scope('Encoder'):
            enc_out = encoder.call(data, noise)

        decoder = Decoder(
            n_nodes=self._params.n_nodes,
            n_features=self._params.n_node_types,
            n_edge_types=self._params.n_edge_types,
            num_latent=self._params.num_latent,
            add_summaries=True
        )

        with tf.variable_scope('Decoder'):
            dec_out = decoder.call(enc_out.latent_vec, self._is_training)

        with tf.name_scope('Losses',
                           values=list(data) + list(enc_out) + list(dec_out)):
            loss = self.make_loss(data, enc_out, dec_out)

        train_op = tf.contrib.layers.optimize_loss(
            loss,
            global_step=tf.train.get_or_create_global_step(),
            learning_rate=self._params.learning_rate,
            optimizer=tf.train.AdamOptimizer(
                learning_rate=self._params.learning_rate),
            summaries=['loss', 'gradient_norm'])

        return {'loss': loss, 'train_op': train_op}

    def _get_num_edge_distribution(self, log_rate):
        # Poisson log likelihood
        dist = tfp.distributions.Poisson(
            log_rate=log_rate,
            name='PoissonNumEdges')
        return dist

    def make_loss(self,
                  inputs: GraphData,
                  encoder_output: EncoderOutputs,
                  decoder_output: DecoderOutputs) -> tf.Tensor:
        # kl divergence
        """
        with tf.name_scope("KL_div", values=[encoder_output.mu, encoder_output.sigma]):
            kl_divs = []
            for i in range(self._params.n_nodes):
                val = layers.kl_div_gaussian_loss(
                    encoder_output.mu[i], encoder_output.sigma[i],
                    scope='kl_div_{}'.format(i))
                kl_divs.append(val)
            kl_divs = tf.add_n(kl_divs)
        """
        kl_divs = layers.kl_div_gaussian_loss(
            encoder_output.mu, encoder_output.log_variance,
            scope='kl_div')

        if self._params.with_masking:
            edge_weight_loss = layers.masked_edge_weights_loss(
                decoder_output.edge_logits,
                inputs.edge_weights, decoder_output.edge_weight_logits,
                log_weight_indicators=inputs.log_weight_indicator,
                true_edge_indices=inputs.edge_index,
                true_edge_list=inputs.edge_list,
                max_edges=self._params.max_edges)
        else:
            edge_weight_loss = layers.edge_weights_loss(
                inputs.edge_weights, decoder_output.edge_weight_logits)

        # node feature logits
        f_labels = tf.argmax(inputs.features, axis=1)
        p_labels = tf.argmax(decoder_output.feature_logits, axis=1)
        feature_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=f_labels,
            logits=decoder_output.feature_logits)
        acc = tf.reduce_mean(tf.to_float(tf.equal(f_labels, p_labels)))
        tf.summary.scalar('feature_accuracy', acc)

        dist = self._get_num_edge_distribution(
            log_rate=decoder_output.num_edges_log_rate)
        num_edges = tf.to_float(inputs.num_edges)
        num_edge_ll = -dist.log_prob(num_edges)
        num_edge_pred = dist.mean()

        resid = tf.cast(inputs.num_edges, num_edge_pred.dtype) - num_edge_pred
        tf.summary.scalar('residuals_num_edges', resid)

        def is_weight(x):
            name = x.name.split('/')[-1]
            return "kernel" in name

        with tf.name_scope('weight_decay'):
            reg = []
            for w in filter(is_weight, tf.trainable_variables()):
                reg.append(tf.nn.l2_loss(w))
            reg = tf.add_n(reg)
            reg *= tf.to_float(self._params.weight_decay)
            tf.summary.scalar('reg_loss', reg)

        tf.summary.scalar('neg_kl_div', kl_divs)
        tf.summary.scalar('feature_loss', feature_loss)
        tf.summary.scalar('num_nodes', self._params.n_nodes)
        tf.summary.scalar('nll_num_edges', num_edge_ll)

        loss = tf.add_n([kl_divs, edge_weight_loss,
                         feature_loss, num_edge_ll, reg])
        return loss


class NeVaeEncoder:

    def __init__(self,
                 num_latent: int,
                 n_layers: int,
                 add_summaries: bool = False) -> None:
        self._num_latent = num_latent
        self._n_layers = n_layers
        self._add_summaries = add_summaries

    def call(self,
             inputs: GraphData,
             noise: tf.Tensor) -> EncoderOutputs:
        with tf.variable_scope('input_layer'):
            outputs = self._gcn_layer(inputs)
            hidden_concat = tf.concat(outputs, axis=1)

        with tf.variable_scope('gating'):
            enc_hidden = self._gating_layer(hidden_concat)

        with tf.variable_scope('inference', values=[enc_hidden]):
            outputs = self._inference_layer(enc_hidden, noise)
        return outputs

    def _inference_layer(self,
                         enc_hidden: tf.Tensor,
                         noise: tf.Tensor) -> EncoderOutputs:
        outputs = layers.dense(enc_hidden, 2 * self._num_latent,
                               activation=tf.nn.tanh,
                               name="dense_1")
        outputs = layers.dense(outputs, 2 * self._num_latent,
                               activation=None,
                               name="dense_2")
        mu, log_variance = tf.split(outputs, num_or_size_splits=2, axis=1)

        with tf.name_scope('reparametrization', values=[mu, log_variance, noise]):
            z = mu + tf.exp(0.5 * log_variance) * noise  # n_nodes x n_latent

            if self._add_summaries:
                tf.summary.histogram('latent_z', z)

        outputs = EncoderOutputs(z, mu, log_variance)
        return outputs

    def _gcn_layer(self, inputs: GraphData) -> List[tf.Tensor]:
        x = inputs.features
        num_outputs = x.get_shape()[-1].value
        x = layers.dense(x, num_outputs, use_bias=False)
        outputs = [x]
        for _ in range(1, self._n_layers):
            x = layers.mlp_neighborhood_aggregation(
                x,
                adjacency=inputs.adjacency,
                use_bias=False)
            outputs.append(x)
        return x

    def _gating_layer(self, hidden_concat: tf.Tensor) -> tf.Tensor:
        return hidden_concat


class Decoder:

    def __init__(self,
                 n_nodes: int,
                 n_features: int,
                 n_edge_types: int,
                 num_latent: int,
                 add_summaries: bool = False) -> None:
        self._n_nodes = n_nodes
        self._n_features = n_features
        self._n_edge_types = n_edge_types
        self._num_latent = num_latent
        self._add_summaries = add_summaries

    def _concat_weights(self,
                        latent_vec: tf.Tensor,
                        n_edge_types: int) -> tf.Tensor:
        """concat node's latent vectors and one-hot encoding of edge type"""
        rows, cols = np.tril_indices(self._n_nodes)

        latent_vec = tf.unstack(latent_vec)
        z_stack_weight = []
        for i, j in zip(rows, cols):
            v = latent_vec[i]
            u = latent_vec[j]
            for k in range(n_edge_types):
                one_hot = tf.one_hot(k, n_edge_types)
                z_stack_weight.append(
                    tf.concat((u, v, one_hot), axis=0))
        weights_in = tf.stack(z_stack_weight)

        return weights_in

    def _concat_edges(self,
                      latent_vec: tf.Tensor) -> tf.Tensor:
        """concat node's latent vectors with a one-hot representation of node type"""
        rows, cols = np.tril_indices(self._n_nodes)

        latent_vec = tf.unstack(latent_vec)
        z_stack_edge = []
        for i, j in zip(rows, cols):
            v = latent_vec[i]
            u = latent_vec[j]
            z_stack_edge.append(
                tf.concat((v, u), axis=0))
        edge_in = tf.stack(z_stack_edge)

        return edge_in

    def _concat_features(self,
                         latent_vec: tf.Tensor,
                         n_features: int) -> tf.Tensor:
        """concat node's latent representation with its feature vector"""
        latent_vec = tf.unstack(latent_vec)
        z_stack_label = []
        for i in range(self._n_nodes):
            v = latent_vec[i]
            for j in range(n_features):
                one_hot = tf.one_hot(j, n_features)
                z_stack_label.append(
                    tf.concat((v, one_hot), axis=0))
        nodes_in = tf.stack(z_stack_label)

        return nodes_in

    def call(self, latent_vec: tf.Tensor, is_training: bool = False) -> DecoderOutputs:
        with tf.name_scope('concat_edges', values=[latent_vec]):
            edge_in = self._concat_edges(latent_vec)
        edge_logits = layers.dense(edge_in, self._num_latent,
                                   activation=tf.nn.tanh,
                                   name='edge_1')
        edge_logits = layers.dense(edge_logits, 1,
                                   activation=None,
                                   use_bias=False,  # bias get's cancelled out in loss
                                   name='edge_2')

        with tf.name_scope('concat_weights', values=[latent_vec]):
            weights_in = self._concat_weights(latent_vec, self._n_edge_types)
        edge_weight_logits = layers.dense(weights_in, self._num_latent,
                                         activation=tf.nn.tanh,
                                         name='weight_1')
        edge_weight_logits = layers.dense(edge_weight_logits, 1,
                                          activation=None,
                                          use_bias=False,  # bias get's cancelled out in loss
                                          name='weight_2')
        edge_weight_logits = tf.reshape(edge_weight_logits, [-1, self._n_edge_types])

        with tf.name_scope('concat_features', values=[latent_vec]):
            nodes_in = self._concat_features(latent_vec, self._n_features)

        node_logits = layers.dense(nodes_in, self._num_latent,
                                   activation=tf.nn.tanh,
                                   name='label_1')
        node_logits = layers.dense(node_logits, 1,
                                   activation=None,
                                   use_bias=False,  # bias get's cancelled out in loss
                                   name='label_2')
        node_logits = tf.reshape(node_logits, [self._n_nodes, self._n_features])

        with tf.variable_scope('poisson_regression'):
            poisson_log_rate = layers.dense(latent_vec, self._n_features,
                                            activation=tf.nn.tanh,
                                            name='dense_1')
            poisson_log_rate = layers.dense(poisson_log_rate, 1,
                                            activation=None,
                                            use_bias=False,
                                            name='dense_2_1')
            out_n_nodes = layers.dense(tf.convert_to_tensor(np.atleast_2d(self._n_nodes), dtype=tf.float32), 1,
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       name='dense_2_2')
            poisson_log_rate += out_n_nodes

            poisson_log_rate = tf.reduce_sum(poisson_log_rate)

        if self._add_summaries:
            tf.summary.histogram('edge_logits', edge_logits)
            tf.summary.histogram('edge_weight_logits', edge_weight_logits)
            tf.summary.histogram('node_logits', node_logits)
            tf.summary.scalar('edges_log_rate', poisson_log_rate)

        return DecoderOutputs(edge_logits, edge_weight_logits, node_logits, poisson_log_rate)


class DecoderWithWhileLoop(Decoder):

    def __init__(self,
                 n_nodes: int,
                 n_features: int,
                 n_edge_types: int,
                 num_latent: int,
                 add_summaries: bool = False) -> None:
        super(DecoderWithWhileLoop, self).__init__(
            n_nodes=n_nodes,
            n_features=n_features,
            n_edge_types=n_edge_types,
            num_latent=num_latent,
            add_summaries=add_summaries
        )
        self._dim_adj_flat = n_nodes * (n_nodes + 1) // 2

    def _concat_weights(self,
                        latent_vec: tf.Tensor,
                        n_edge_types: int) -> tf.Tensor:
        """concat node's latent vectors and one-hot encoding of edge type"""
        rows, cols = map(tf.convert_to_tensor,
                         np.tril_indices(self._n_nodes))

        def weights_cond(index, k, array):
            return index < rows.shape[0]

        def weights_body(index, k, out_array):
            i = rows[index]
            j = cols[index]
            v = latent_vec[i]
            u = latent_vec[j]
            one_hot = tf.one_hot(k, n_edge_types)
            values = tf.concat((u, v, one_hot), axis=0)
            ret_array = out_array.write(index * n_edge_types + k, values)

            next_k = k + 1
            ret_k = tf.floormod(next_k, n_edge_types)
            ret_index = index + tf.floor_div(next_k, n_edge_types)
            return ret_index, ret_k, ret_array

        arr_stack_weight = tf.TensorArray(
            tf.float32,
            size=self._dim_adj_flat * n_edge_types,
            element_shape=tf.TensorShape([latent_vec.shape[1].value * 2 + n_edge_types]),
            clear_after_read=False
        )
        out = tf.while_loop(
            cond=weights_cond,
            body=weights_body,
            loop_vars=(0, 0, arr_stack_weight),
            parallel_iterations=1,
            return_same_structure=True)
        weights_in = out[-1].stack()
        weights_in.set_shape([self._dim_adj_flat * n_edge_types, weights_in.shape[1].value])
        return weights_in

    def _concat_edges(self,
                      latent_vec: tf.Tensor) -> tf.Tensor:
        """concat node's latent vectors with a one-hot representation of node type"""
        rows, cols = map(tf.convert_to_tensor,
                         np.tril_indices(self._n_nodes))

        def edges_cond(index, array):
            return index < rows.shape[0]

        def edges_body(index, out_array):
            i = rows[index]
            j = cols[index]
            v = latent_vec[i]
            u = latent_vec[j]
            values = tf.concat((v, u), axis=0)
            ret_array = out_array.write(index, values)
            return index + 1, ret_array

        arr_stack_edge = tf.TensorArray(
            tf.float32,
            size=self._dim_adj_flat,
            element_shape=tf.TensorShape([latent_vec.shape[1].value * 2]),
            clear_after_read=False
        )
        out = tf.while_loop(
            cond=edges_cond,
            body=edges_body,
            loop_vars=(0, arr_stack_edge),
            parallel_iterations=1,
            return_same_structure=True)
        edge_in = out[-1].stack()
        edge_in.set_shape([self._dim_adj_flat, edge_in.shape[1].value])
        return edge_in

    def _concat_features(self,
                         latent_vec: tf.Tensor,
                         n_features: int) -> tf.Tensor:
        """concat node's latent representation with its feature vector"""
        shape = (self._n_nodes, n_features)

        def features_cond(index, array):
            return index < np.prod(shape)

        def features_body(index, out_array):
            i, j = unravel_index(index, shape)
            v = latent_vec[i]
            one_hot = tf.one_hot(j, n_features)
            values = tf.concat((v, one_hot), axis=0)
            ret_array = out_array.write(index, values)
            return index + 1, ret_array

        arr_stack_feature = tf.TensorArray(
            tf.float32,
            size=self._n_nodes * n_features,
            element_shape=tf.TensorShape([latent_vec.shape[1].value + n_features]),
            clear_after_read=False
        )
        out = tf.while_loop(
            cond=features_cond,
            body=features_body,
            loop_vars=(0, arr_stack_feature),
            parallel_iterations=1,
            return_same_structure=True)
        nodes_in = out[-1].stack()
        nodes_in.set_shape([self._n_nodes * n_features, nodes_in.shape[1].value])
        return nodes_in
