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
import logging
from typing import Callable, Dict, Optional, Tuple
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ..base import MolGANModel, MolGANLoss
from ..estimator import make_eval_estimator_spec, make_train_estimator_spec
from ..losses import non_saturating_loss, reward_loss
from ..molgan import MolGANEstimator, MolGANParameters
from ..regularization import gradient_penalty, _penalty_from_gradient, variance_penalty

LOG = logging.getLogger(__name__)

TensorDict = Dict[str, tf.Tensor]
EncoderFn = Callable[[TensorDict], tf.Tensor]
DiscriminatorFn = Callable[[TensorDict, tf.Tensor], tf.Tensor]
CycleDiscriminatorFn = Callable[[TensorDict, TensorDict], tf.Tensor]
UnaryDiscriminatorFn = Callable[[TensorDict], tf.Tensor]
GeneratorFn = Callable[[tf.Tensor], Tuple[TensorDict, TensorDict]]
ValueFn = Optional[Callable[[tf.Tensor], tf.Tensor]]


def _split_inputs(inputs: TensorDict) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return inputs['adjacency_in'], inputs['features'], inputs['embedding']


def make_train_eval_model(real_data: TensorDict,
                          embedding: tf.Tensor,
                          generator_fn: GeneratorFn,
                          encoder_fn: EncoderFn,
                          discriminator_fn: DiscriminatorFn,
                          cycle_discriminator_fn: Optional[CycleDiscriminatorFn],
                          unary_discriminator_fn: Optional[UnaryDiscriminatorFn],
                          value_fn: ValueFn) -> Tuple[MolGANModel, TensorDict]:
    with tf.variable_scope('Generator') as gen_scope:
        generated_data, predicted_data = generator_fn(embedding)
        with tf.variable_scope('Encoder'):
            encoder_real_outputs = encoder_fn(real_data)
    with tf.variable_scope(gen_scope, reuse=True):
        reconstructed_data, _ = generator_fn(encoder_real_outputs)

    for real_name in ('adjacency_in', 'features'):
        real_val = real_data[real_name]
        gen_val = generated_data[real_name]
        if not gen_val.shape.is_compatible_with(real_val.shape):
            raise ValueError(
                'Generator output %s shape (%s) must be the same shape as real data '
                '(%s).' % (real_name, gen_val.shape, real_val.shape))
        recon_val = reconstructed_data[real_name]
        if not recon_val.shape.is_compatible_with(real_val.shape):
            raise ValueError(
                'Reconstructed output %s shape (%s) must be the same shape as real data '
                '(%s).' % (real_name, recon_val.shape, real_val.shape))

    if not encoder_real_outputs.shape.is_compatible_with(embedding.shape):
        raise ValueError(
            'Encoded latent space shape (%s) must be the same shape as '
            'generator latent space (%s).' % (encoder_real_outputs.shape, embedding.shape))
    generated_data['embedding'] = encoder_real_outputs

    with tf.variable_scope('Discriminator') as dis_scope:
        with tf.variable_scope('Joint') as dis_xz_scope:
            discriminator_gen_outputs = discriminator_fn(generated_data, embedding)
        with tf.variable_scope(dis_xz_scope, reuse=True):
            discriminator_real_outputs = discriminator_fn(real_data, encoder_real_outputs)

        if cycle_discriminator_fn is not None:
            with tf.variable_scope('Cycle') as diss_x_scope:
                diss_cycle_gen_outputs = cycle_discriminator_fn(real_data, reconstructed_data)
            with tf.variable_scope(diss_x_scope, reuse=True):
                diss_cycle_real_outputs = cycle_discriminator_fn(real_data, real_data)
        else:
            diss_cycle_gen_outputs = None
            diss_cycle_real_outputs = None

        if unary_discriminator_fn is not None:
            with tf.variable_scope('Unary') as diss_unary_scope:
                diss_unary_gen_outputs = unary_discriminator_fn(generated_data)
            with tf.variable_scope(diss_unary_scope, reuse=True):
                diss_unary_real_outputs = unary_discriminator_fn(real_data)
        else:
            diss_unary_gen_outputs = None
            diss_unary_real_outputs = None

    # Get model-specific variables.
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope.name)
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dis_scope.name)

    if value_fn is None:
        value_gen_outputs = None
        value_real_outputs = None
        value_variables = None
        value_scope = None
    else:
        with tf.variable_scope('ValueNet') as value_scope:
            value_gen_outputs = value_fn(generated_data)
        with tf.variable_scope(value_scope, reuse=True):
            value_real_outputs = value_fn(real_data)

        value_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=value_scope.name)

    gan_model = MolGANModel(
        embedding,
        generated_data,
        generator_variables,
        gen_scope,
        generator_fn,
        real_data,
        (discriminator_real_outputs, diss_cycle_real_outputs, diss_unary_real_outputs),
        (discriminator_gen_outputs, diss_cycle_gen_outputs, diss_unary_gen_outputs),
        discriminator_variables,
        dis_scope,
        discriminator_fn,
        value_real_outputs,
        value_gen_outputs,
        value_variables,
        value_scope
    )
    return gan_model, predicted_data


def make_prediction_model(generator_inputs: tf.Tensor,
                          real_data:  TensorDict,
                          generator_fn: GeneratorFn,
                          encoder_fn: EncoderFn,
                          value_fn: ValueFn) -> MolGANModel:
    has_decoder = generator_inputs is not None
    has_encoder = real_data is not None

    if not has_decoder and not has_encoder:
        raise ValueError('at least one of real_data and generator_inputs must be provided')

    with tf.variable_scope('Generator') as gen_scope:
        if has_decoder:
            generated_data, predicted_data = generator_fn(generator_inputs)
        else:
            LOG.info('Creating prediction model without generator/decoder.')
            generated_data = {}
            predicted_data = {}

        if has_encoder:
            with tf.variable_scope('Encoder'):
                encoder_real_outputs = encoder_fn(real_data)
        else:
            LOG.info('Creating prediction model without encoder.')

    if has_encoder:
        with tf.variable_scope('Generator', reuse=has_decoder):
            _, reconstructed_data = generator_fn(encoder_real_outputs)

        for key, value in reconstructed_data.items():
            predicted_data['reconstructed/{}'.format(key)] = value
        predicted_data['embedding'] = encoder_real_outputs

    value_gen_outputs = None
    if value_fn is not None and has_decoder:
        with tf.variable_scope('ValueNet'):
            value_gen_outputs = value_fn(generated_data)

    return MolGANModel(
        generator_inputs,
        predicted_data,
        None,  # generator_variables
        gen_scope,
        generator_fn,
        real_data=None,
        discriminator_real_outputs=None,
        discriminator_gen_outputs=None,
        discriminator_variables=None,
        discriminator_scope=None,
        discriminator_fn=None,
        valuenet_real_probs=None,
        valuenet_gen_probs=value_gen_outputs,
        valuenet_variables=None,
        valuenet_scope=None
    )


def non_saturating_encoder_loss(model: MolGANModel,
                                add_summaries: bool = False,
                                scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'EncoderLoss',
                       [model.discriminator_gen_outputs]):
        loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.discriminator_real_outputs,
                labels=tf.zeros_like(model.discriminator_real_outputs)))
        tf.losses.add_loss(loss_G)
        if add_summaries:
            tf.summary.scalar('encoder_gan_loss', loss_G)
    return loss_G


def ali_gradient_penalty(model: MolGANModel,
                         real_adj: tf.Tensor,
                         real_features: tf.Tensor,
                         real_embedding: tf.Tensor,
                         gen_adj: tf.Tensor,
                         gen_features: tf.Tensor,
                         gen_embedding: tf.Tensor,
                         gradient_penalty_weight: float = 10.0,
                         target: float = 1.0,
                         add_summaries: bool = False,
                         scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'GradientPenalty',
                       [real_adj, real_features, gen_adj, gen_features]):
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

        eps_embedding = eps
        for _ in range(real_features.shape.ndims - 1):
            eps_embedding = tf.expand_dims(eps_embedding, axis=-1)

        intp_embedding = eps_embedding * real_embedding + (1.0 - eps_embedding) * gen_embedding

        with tf.name_scope(None):  # Clear scope so update ops are added properly.
            with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
                disc_interpolates = model.discriminator_fn(intp_inputs, intp_embedding,
                                                           mode=tf.estimator.ModeKeys.TRAIN)

        grads = tf.gradients(disc_interpolates,
                             (intp_inputs['adjacency_in'], intp_inputs['features'], intp_embedding))

        penalty = tf.reduce_sum([_penalty_from_gradient(g, batch_size, target=target, scope=sc)
                                 for g, sc in zip(grads, ['adj_grad_pen', 'feat_grad_pen', 'embedding_grad_pen'])])
        gp_weight = tf.convert_to_tensor(gradient_penalty_weight, dtype=penalty.dtype)
        gp_weight = tf.assert_scalar(gp_weight)
        penalty *= gp_weight

        tf.losses.add_loss(penalty)
        if add_summaries:
            tf.summary.scalar('gradient_penalty_loss', penalty)

    return penalty


class MolGANALICEEstimator(MolGANEstimator):

    def __init__(self,
                 hooks_fn,
                 params: MolGANParameters) -> None:
        super(MolGANALICEEstimator, self).__init__(hooks_fn=hooks_fn, params=params)

    def _stochastic_dense_generator_net(self, inputs, noise):
        x = inputs
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.tanh):
            if self.params.without_generator_skip_connections:
                x = tf.concat([noise, x], axis=1)
                for i, units in enumerate(self.params.generator_units):
                    x = slim.fully_connected(x, units, scope='dense{}'.format(i))
            else:
                xs = tf.split(x, len(self.params.generator_units), axis=1)
                for i, units in enumerate(self.params.generator_units):
                    if i == 0:
                        x = tf.concat([noise, xs[i]], axis=1)
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

    def generator_fn(self,
                     inputs: tf.Tensor,
                     noise: tf.Tensor,
                     mode: str):
        logits_adj, logits_features = self._stochastic_dense_generator_net(inputs, noise)

        samples, predictions = self._generator_softmax(logits_adj, logits_features)

        self._append_generator_penalties(samples, mode)

        # zero edge type corresponds to absent edges
        outputs = {'adjacency_in': samples['adjacency'][:, 1:, :, :],
                   'features': samples['features']}
        return outputs, predictions

    def _graph_conv_net(self,
                        inputs:  TensorDict,
                        embedding: tf.Tensor,
                        mode: str) -> tf.Tensor:
        emb_shape = embedding.shape[0]
        for k, v in inputs.items():
            if not emb_shape.is_compatible_with(v.shape[0]):
                raise ValueError('Shapes %s and %s (%s) are incompatible' % (emb_shape, v.shape[0], k))

        with tf.variable_scope('GraphConv'):
            net_1 = self._discriminator_net(inputs, mode=mode)

        with tf.variable_scope('Embedding'):
            net_2 = embedding
            units = 4 * self.params.discriminator_dense_units[-1]
            net_2 = slim.fully_connected(net_2, units,
                                         activation_fn=tf.nn.tanh)

        net = tf.concat([net_1, net_2], 1)
        return net

    def discriminator_fn(self,
                         inputs: TensorDict,
                         embedding: tf.Tensor,
                         mode: str) -> tf.Tensor:
        net = self._graph_conv_net(inputs, embedding, mode)
        # don't need bias, gets canceled out in Wasserstein loss
        outputs = slim.linear(net, 1, biases_initializer=None)
        return outputs

    def unary_discriminator_fn(self,
                               inputs,
                               mode):
        x = self._discriminator_net(inputs, mode)
        # don't need bias, gets canceled out in Wasserstein loss
        outputs = slim.linear(x, 1, biases_initializer=None)
        return outputs

    def encoder_fn(self,
                   inputs: TensorDict,
                   noise: tf.Tensor,
                   mode: str) -> tf.Tensor:
        net = self._graph_conv_net(inputs, noise, mode)
        outputs = slim.linear(net, self.params.num_latent)
        return outputs

    def cycle_discriminator_fn(self,
                               inputs_1: TensorDict,
                               inputs_2: TensorDict,
                               mode: str) -> tf.Tensor:
        with tf.variable_scope('Stack_1'):
            net_1 = self._discriminator_net(inputs_1, mode=mode)
        with tf.variable_scope('Stack_2'):
            net_2 = self._discriminator_net(inputs_2, mode=mode)

        net = tf.multiply(net_1, net_2)
        net = slim.fully_connected(net, self.params.discriminator_dense_units[-1],
                                   activation_fn=tf.nn.tanh)
        net = slim.linear(net, 1)
        return net

    def _make_loss(self,
                   model: MolGANModel,
                   mode: str) -> MolGANLoss:
        xz_gen_outputs, xx_gen_outputs, x_gen_outputs = model.discriminator_gen_outputs
        xz_real_outputs, xx_real_outputs, x_real_outputs = model.discriminator_real_outputs
        add_summaries = self.params.add_summaries

        loss_D = []
        loss_G = []
        with tf.name_scope('Losses'):
            xz_model = model._replace(discriminator_gen_outputs=xz_gen_outputs,
                                      discriminator_real_outputs=xz_real_outputs)
            xz_loss = non_saturating_loss(xz_model, add_summaries=add_summaries,
                                          scope='Joint')

            xz_enc = non_saturating_encoder_loss(xz_model, add_summaries=add_summaries,
                                                 scope='Joint')

            loss_D.append(xz_loss.discriminator_loss)
            loss_G.extend([xz_loss.generator_loss, xz_enc])

            if xx_gen_outputs is not None:
                xx_model = model._replace(discriminator_gen_outputs=xx_gen_outputs,
                                        discriminator_real_outputs=xx_real_outputs)
                xx_loss = non_saturating_loss(xx_model, add_summaries=add_summaries,
                                              scope='Cycle')

                xx_enc = non_saturating_encoder_loss(xx_model, add_summaries=add_summaries,
                                                     scope='Cycle')

                loss_D.append(xx_loss.discriminator_loss)
                loss_G.extend([xx_loss.generator_loss, xx_enc])

            if x_gen_outputs is not None:
                x_model = model._replace(discriminator_gen_outputs=x_gen_outputs,
                                         discriminator_real_outputs=x_real_outputs)
                x_loss = non_saturating_loss(x_model, add_summaries=add_summaries,
                                             scope='Unary')

                loss_D.append(x_loss.discriminator_loss)
                loss_G.append(x_loss.generator_loss)

            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.params.gradient_penalty_weight > 0:
                    gp = self._get_gradient_penalties(
                        model, add_summaries=add_summaries)
                    loss_D.extend(gp)

                if self.params.variance_penalty_weight != 0:
                    pen = variance_penalty(xz_model.discriminator_gen_outputs,
                                           self.params.variance_penalty_weight,
                                           add_summaries=add_summaries)
                    self._generator_penalty.append(pen)

        if self.params.with_value_net:
            r_loss = reward_loss(model,
                                 model.real_data['reward_real'],
                                 model.real_data['reward_generated'],
                                 xz_loss,
                                 self.params.lam,
                                 add_summaries=add_summaries)
            # update generator xz loss
            loss_G[0] = r_loss.generator_loss
            loss_V = r_loss.valuenet_loss
        else:
            loss_V = None

        molgan_loss = MolGANLoss(
            generator_loss=tf.add_n(loss_G + self._generator_penalty),
            discriminator_loss=tf.add_n(loss_D),
            valuenet_loss=loss_V)

        return molgan_loss

    def _get_gradient_penalties(self,
                                gan_model: MolGANModel,
                                add_summaries: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # cycle discriminator
        real_adj, real_features, real_embedding = _split_inputs(gan_model.real_data)
        gen_adj, gen_features, gen_embedding = _split_inputs(gan_model.generated_data)

        def wrap_cycle__discriminator_fn(x, **kwargs):
            return self.cycle_discriminator_fn(inputs_1=gan_model.real_data,
                                               inputs_2=x, **kwargs)

        if not self.params.without_cycle_discriminator:
            cyc_model = gan_model._replace(discriminator_fn=wrap_cycle__discriminator_fn)

            cyc_penalty = gradient_penalty(cyc_model,
                                           real_adj, real_features,
                                           gen_adj, gen_features,
                                           discriminator_scope='Discriminator/Cycle',
                                           gradient_penalty_weight=self.params.gradient_penalty_weight,
                                           add_summaries=add_summaries)
        else:
            cyc_penalty = tf.zeros([])

        # joint discriminator
        joint_model = gan_model._replace(
            discriminator_fn=partial(self.discriminator_fn, embedding=real_embedding))

        joint_penalty = gradient_penalty(joint_model,
                                         real_adj, real_features,
                                         gen_adj, gen_features,
                                         discriminator_scope='Discriminator/Joint',
                                         gradient_penalty_weight=self.params.gradient_penalty_weight,
                                         add_summaries=add_summaries)

        if not self.params.without_unary_discriminator:
            unary_model = gan_model._replace(
                discriminator_fn=partial(self.unary_discriminator_fn))

            unary_penalty = gradient_penalty(unary_model,
                                             real_adj, real_features,
                                             gen_adj, gen_features,
                                             discriminator_scope='Discriminator/Unary',
                                             gradient_penalty_weight=self.params.gradient_penalty_weight,
                                             add_summaries=add_summaries)
        else:
            unary_penalty = tf.zeros([])

        """
        joint_model = gan_model._replace(discriminator_fn=self.discriminator_fn)

        joint_penalty = ali_gradient_penalty(joint_model,
                                             real_adj, real_features, real_embedding,
                                             gen_adj, gen_features, gen_embedding,
                                             self.params.gradient_penalty_weight,
                                             add_summaries=add_summaries,
                                             scope='JointGradientPenalty')
        """
        return cyc_penalty, joint_penalty, unary_penalty

    def model_fn(self,
                 features: tf.Tensor,
                 labels: TensorDict,
                 mode: str) -> tf.estimator.EstimatorSpec:
        real_data = labels  # rename inputs for clarity
        generator_inputs = features  # rename inputs for clarity

        discriminator_fn: DiscriminatorFn = partial(self.discriminator_fn, mode=mode)
        if self.params.without_cycle_discriminator:
            cycle_discriminator_fn = None
        else:
            cycle_discriminator_fn: CycleDiscriminatorFn = partial(self.cycle_discriminator_fn, mode=mode)
        if self.params.without_unary_discriminator:
            unary_discriminator_fn = None
        else:
            unary_discriminator_fn: UnaryDiscriminatorFn = partial(self.unary_discriminator_fn, mode=mode)
        value_fn: ValueFn = partial(self.value_fn, mode=mode) if self.params.with_value_net else None
        self._generator_penalty = []

        if mode == tf.estimator.ModeKeys.PREDICT:
            if 'adjacency_in' in generator_inputs and 'features' in generator_inputs:
                real_data = {'adjacency_in': generator_inputs['adjacency_in'],
                             'features': generator_inputs['features']}
                real_data = self._one_hot_encoding(real_data)
            else:
                real_data = None

            if 'embedding' in generator_inputs:
                embedding = generator_inputs['embedding']
            else:
                embedding = None

            generator_fn: GeneratorFn = partial(
                self.generator_fn, noise=generator_inputs['noise'], mode=mode)
            encoder_fn: DiscriminatorFn = partial(
                self.encoder_fn, noise=generator_inputs['noise'], mode=mode)

            gan_model = make_prediction_model(embedding, real_data,
                                              generator_fn, encoder_fn, value_fn)
            predictions = gan_model.generated_data
            if self.params.with_value_net:
                predictions['reward'] = gan_model.valuenet_gen_probs
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
        else:
            self._valence_tensor = real_data.pop('valence')
            embedding = real_data['embedding']
            encoder_fn = partial(self.encoder_fn, noise=generator_inputs, mode=mode)
            generator_fn = partial(self.generator_fn, noise=generator_inputs, mode=mode)
            real_data = self._one_hot_encoding(real_data)

            gan_model, predicted_data = make_train_eval_model(
                real_data,
                embedding,
                generator_fn,
                encoder_fn,
                discriminator_fn,
                cycle_discriminator_fn,
                unary_discriminator_fn,
                value_fn)

            molgan_loss = self._make_loss(
                gan_model,
                mode)

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
