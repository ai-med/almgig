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
from functools import partial
from typing import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.gan as tfgan

from ..gcn_layers import relational_gin_mlp_neighborhood_aggregation, graph_aggregation_layer_with_gate
from .base import MolGANLoss, MolGANModel, make_prediction_model, make_train_eval_model
from .estimator import make_eval_estimator_spec, make_train_estimator_spec, MolGANTrainHooksFn
from .losses import wasserstein_loss, reward_loss
from .regularization import gradient_penalty, connectivity_penalty, valence_penalty, variance_penalty


@slim.add_arg_scope
def gumbel_softmax(inputs: tf.Tensor,
                   temperature: float = 1.0,
                   symmetric: bool = False,
                   axis: int = -1,
                   scope: Optional[str] = None) -> tf.Tensor:
    if inputs.shape[axis].value <= 2:
        raise ValueError('logits must have at least size 3 on axis={}'.format(axis))

    temperature = tf.convert_to_tensor(temperature, dtype=inputs.dtype)
    with tf.variable_scope(scope, 'gumbel_softmax', [inputs]):
        temperature = tf.assert_scalar(temperature)
        assert_op = tf.assert_greater(temperature, 0.0)

        with tf.control_dependencies([assert_op]):
            gumbel = -tf.log(-tf.log(tf.random_uniform(inputs.shape, dtype=inputs.dtype)))
            if symmetric:
                gumbel = (gumbel + tf.matrix_transpose(gumbel)) / 2.0
            gumbel_logits = gumbel + inputs
            gumbel_softmax = tf.exp(tf.nn.log_softmax(tf.div(gumbel_logits, temperature), axis=axis))

    return gumbel_softmax


def argmax_one_hot(inputs: tf.Tensor, axis: int) -> tf.Tensor:
    discrete_argmax = tf.argmax(inputs, axis=axis)
    discrete_one_hot = tf.one_hot(discrete_argmax,
                                  depth=inputs.shape[axis],
                                  dtype=inputs.dtype,
                                  axis=axis)
    return discrete_one_hot


def _split_inputs(inputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    return inputs['adjacency_in'], inputs['features']


class MolGANParameters:

    def __init__(self,
                 max_nodes: int,
                 num_node_types: int,
                 num_edge_types: int,
                 num_latent: int,
                 with_value_net: bool,
                 lam: float,
                 generator_learning_rate: float,
                 discriminator_learning_rate: float,
                 beta1: float,
                 beta2: float,
                 gradient_penalty_weight: float,
                 connectivity_penalty_weight: float,
                 valence_penalty_weight: float,
                 variance_penalty_weight: float,
                 batch_size: int,
                 discriminator_gcn_units: Sequence[int],
                 discriminator_dense_units: Sequence[int],
                 generator_units: Sequence[int],
                 without_cycle_discriminator: bool = False,
                 without_unary_discriminator: bool = False,
                 without_gcn_skip_connections: bool = False,
                 without_generator_skip_connections: bool = False,
                 without_gated_gcn: bool = False,
                 softmax_temp: float = 1.0,
                 add_summaries: bool = False,
                 summarize_gradients: bool = False) -> None:
        self.max_nodes = max_nodes
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_latent = num_latent
        self.with_value_net = with_value_net
        self.lam = lam
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_penalty_weight = gradient_penalty_weight
        self.connectivity_penalty_weight = connectivity_penalty_weight
        self.valence_penalty_weight = valence_penalty_weight
        self.variance_penalty_weight = variance_penalty_weight
        self.batch_size = batch_size
        self.discriminator_gcn_units = discriminator_gcn_units
        self.discriminator_dense_units = discriminator_dense_units
        self.generator_units = generator_units
        self.without_cycle_discriminator = without_cycle_discriminator
        self.without_unary_discriminator = without_unary_discriminator
        self.without_gcn_skip_connections = without_gcn_skip_connections
        self.without_generator_skip_connections = without_generator_skip_connections
        self.without_gated_gcn = without_gated_gcn
        self.softmax_temp = softmax_temp
        self.add_summaries = add_summaries
        self.summarize_gradients = summarize_gradients


class MolGANEstimator:

    def __init__(self,
                 hooks_fn: MolGANTrainHooksFn,
                 params: MolGANParameters) -> None:
        self.hooks_fn = hooks_fn
        self.params = params
        self._generator_penalty = None
        self._valence_tensor = None

    def _dense_generator_net(self, inputs):
        x = inputs
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.tanh):
            if self.params.without_generator_skip_connections:
                for i, units in enumerate(self.params.generator_units):
                    x = slim.fully_connected(x, units, scope='dense{}'.format(i))
            else:
                xs = tf.split(x, len(self.params.generator_units), axis=1)
                for i, units in enumerate(self.params.generator_units):
                    if i == 0:
                        x = xs[i]
                    else:
                        x = tf.concat([x, xs[i]], axis=1)
                    x = slim.fully_connected(x, units, scope='dense{}'.format(i))

            max_nodes = self.params.max_nodes
            with tf.variable_scope('logits_adj', values=[x]):
                # bond_type : batch_size x num_bond_types x max_nodes x max_nodes
                logits_adj = slim.linear(x,
                                         num_outputs=(self.params.num_edge_types + 1) * max_nodes * max_nodes)
                logits_adj = tf.reshape(logits_adj, [-1, self.params.num_edge_types + 1, max_nodes, max_nodes])
                # make adjacency matrix symmetric
                logits_adj = (logits_adj + tf.matrix_transpose(logits_adj)) / 2.0

            with tf.variable_scope('logits_features', values=[x]):
                logits_features = slim.linear(x, num_outputs=max_nodes * (self.params.num_node_types + 1))
                # node_features : batch_size x max_nodes x n_features (num_atom_types)
                logits_features = tf.reshape(logits_features, [-1, max_nodes, self.params.num_node_types + 1])

        return logits_adj, logits_features

    def _generator_softmax(self, logits_adj, logits_features):
        adj_sampled = gumbel_softmax(logits_adj,
                                     temperature=self.params.softmax_temp,
                                     symmetric=True,
                                     axis=1,
                                     scope='softmax_adj')
        adj_pred = argmax_one_hot(logits_adj, axis=1)
        features_sampled = gumbel_softmax(logits_features,
                                          temperature=self.params.softmax_temp,
                                          axis=-1,
                                          scope='softmax_features')
        features_pred = argmax_one_hot(logits_features, axis=-1)

        # zero edge type corresponds to absent edges
        outputs = {'adjacency': adj_sampled,
                   'features': features_sampled}
        predictions = {'adjacency': tf.identity(adj_pred, name='discrete_adjacency'),
                       'features': tf.identity(features_pred, name='discrete_features')}
        return outputs, predictions

    def _append_generator_penalties(self, samples, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.params.valence_penalty_weight > 0:
                pen = valence_penalty(samples['adjacency'],
                                      samples['features'],
                                      self.params.batch_size,
                                      self._valence_tensor,
                                      self.params.valence_penalty_weight,
                                      add_summaries=self.params.add_summaries)
                self._generator_penalty.append(pen)

            if self.params.connectivity_penalty_weight > 0:
                pen = connectivity_penalty(samples['adjacency'],
                                           samples['features'],
                                           self.params.batch_size,
                                           self.params.connectivity_penalty_weight,
                                           add_summaries=self.params.add_summaries)
                self._generator_penalty.append(pen)

    def generator_fn(self,
                     inputs, mode):
        logits_adj, logits_features = self._dense_generator_net(inputs)

        samples, predictions = self._generator_softmax(logits_adj, logits_features)

        self._append_generator_penalties(samples, mode)

        # zero edge type corresponds to absent edges
        outputs = {'adjacency_in': samples['adjacency'][:, 1:, :, :],
                   'features': samples['features']}
        return outputs, predictions

    def _discriminator_net(self, inputs, mode):
        adj, features = _split_inputs(inputs)

        with slim.arg_scope([relational_gin_mlp_neighborhood_aggregation],
                            adjacency=adj, layers=1, activation_fn=tf.nn.tanh):
            x = features
            layer_outputs = [features]
            for i, units in enumerate(self.params.discriminator_gcn_units):
                x = relational_gin_mlp_neighborhood_aggregation(
                    x, units, scope='gconv_{}'.format(i + 1))
                layer_outputs.append(x)

            with tf.variable_scope('graph_pooling', values=layer_outputs):
                if self.params.without_gcn_skip_connections:
                    layer_outputs = layer_outputs[-1]
                else:
                    layer_outputs = tf.concat(layer_outputs, axis=-1)
                output = slim.fully_connected(layer_outputs,
                                              num_outputs=self.params.discriminator_dense_units[0],
                                              activation_fn=tf.nn.tanh)
                if self.params.without_gated_gcn:
                    # sum over node dimension
                    graph_embedding = tf.reduce_sum(output, axis=1)
                else:
                    graph_embedding = graph_aggregation_layer_with_gate(
                        output,
                        axis=1)

            x = graph_embedding
            for i, units in enumerate(self.params.discriminator_dense_units):
                x = slim.fully_connected(x, units,
                                         activation_fn=tf.nn.tanh,
                                         scope='fc_{}'.format(i + 1))
        return x

    def discriminator_fn(self,
                         inputs,
                         mode):
        x = self._discriminator_net(inputs, mode)
        # don't need bias, gets canceled out in Wasserstein loss
        outputs = slim.linear(x, 1, biases_initializer=None)
        return outputs

    def value_fn(self, inputs, mode):
        x = self._discriminator_net(inputs, mode)
        outputs = slim.fully_connected(x, 1, activation_fn=tf.nn.sigmoid)
        return outputs

    def _one_hot_encoding(self, real_data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        adjacency, features = _split_inputs(real_data)
        with tf.name_scope('one_hot_encoding', values=[adjacency, features]):
            assert adjacency.shape.ndims == 3, 'adjacency.ndims = {} != 3'.format(adjacency.shape.ndims)
            assert features.shape.ndims == 2, 'features.ndims = {} != 2'.format(features.shape.ndims)

            adj_depth = self.params.num_edge_types + 1  # bond types + 1 no-edge type
            adj = tf.one_hot(adjacency, depth=adj_depth, axis=1)  # (batch_dim, edge_type, node_1, node_2)
            adj = adj[:, 1:, :, :]  # zero edge type corresponds to absent edges
            real_data['adjacency_in'] = adj

            node_depth = self.params.num_node_types + 1  # atom types + 1 no-atom type
            feat = tf.one_hot(features, depth=node_depth)
            real_data['features'] = feat
        return real_data

    def _make_loss(self,
                   model: MolGANModel,
                   mode: str) -> MolGANLoss:
        gan_loss = wasserstein_loss(
            model,
            add_summaries=self.params.add_summaries)

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.params.variance_penalty_weight != 0:
                pen = variance_penalty(model.discriminator_gen_outputs,
                                       self.params.variance_penalty_weight,
                                       add_summaries=self.params.add_summaries)
                self._generator_penalty.append(pen)

            loss_D = gan_loss.discriminator_loss
            if self.params.gradient_penalty_weight > 0:
                real_adj, real_features = _split_inputs(model.real_data)
                gen_adj, gen_features = _split_inputs(model.generated_data)
                penalty = gradient_penalty(model,
                                           real_adj, real_features,
                                           gen_adj, gen_features,
                                           discriminator_scope='Discriminator',
                                           gradient_penalty_weight=self.params.gradient_penalty_weight,
                                           add_summaries=self.params.add_summaries)
                loss_D += penalty

            loss_G = gan_loss.generator_loss
            if len(self._generator_penalty) > 0:
                loss_G = tf.add_n([loss_G] + self._generator_penalty)

            gan_loss = tfgan.GANLoss(generator_loss=loss_G,
                                     discriminator_loss=loss_D)

        if self.params.with_value_net:
            molgan_loss = reward_loss(model,
                                      model.real_data['reward_real'],
                                      model.real_data['reward_generated'],
                                      gan_loss,
                                      self.params.lam,
                                      add_summaries=self.params.add_summaries)
        else:
            molgan_loss = MolGANLoss(
                generator_loss=gan_loss.generator_loss,
                discriminator_loss=gan_loss.discriminator_loss,
                valuenet_loss=None)

        return molgan_loss

    def model_fn(self,
                 features: tf.Tensor,
                 labels: Dict[str, tf.Tensor],
                 mode: str) -> tf.estimator.EstimatorSpec:
        real_data = labels  # rename inputs for clarity
        generator_inputs = features  # rename inputs for clarity

        generator_fn = partial(self.generator_fn, mode=mode)
        discriminator_fn = partial(self.discriminator_fn, mode=mode)
        value_fn = partial(self.value_fn, mode=mode) if self.params.with_value_net else None
        self._generator_penalty = []

        # see tensorflow.contrib.gan.python.estimator.python.gan_estimator_impl._get_estimator_spec
        if mode == tf.estimator.ModeKeys.PREDICT:
            gan_model = make_prediction_model(generator_inputs, generator_fn, value_fn)
            predictions = gan_model.generated_data
            if self.params.with_value_net:
                predictions['reward'] = gan_model.valuenet_gen_probs
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
        else:
            self._valence_tensor = real_data.pop('valence')
            real_data = self._one_hot_encoding(real_data)
            gan_model, predicted_data = make_train_eval_model(
                real_data,
                generator_inputs,
                generator_fn,
                discriminator_fn,
                value_fn)

            molgan_loss = self._make_loss(gan_model, mode)

            if mode == tf.estimator.ModeKeys.TRAIN:
                spec = make_train_estimator_spec(gan_model, molgan_loss,
                                                 self.hooks_fn,
                                                 self.params.generator_learning_rate,
                                                 self.params.discriminator_learning_rate,
                                                 self.params.beta1,
                                                 self.params.beta2,
                                                 self.params.add_summaries,
                                                 self.params.summarize_gradients)
            elif mode == tf.estimator.ModeKeys.EVAL:
                gan_model = gan_model._replace(generated_data=predicted_data)
                spec = make_eval_estimator_spec(gan_model, molgan_loss)
            else:
                raise ValueError('mode={!r} is not supported'.format(mode))

        return spec
