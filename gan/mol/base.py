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
from typing import Dict, Tuple
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.gan as tfgan


class MolGANModel(
    namedtuple('MolGANModel', tfgan.GANModel._fields + (
        'valuenet_real_probs',
        'valuenet_gen_probs',
        'valuenet_variables',
        'valuenet_scope'
    ))):
    """An MolGANModel contains all the pieces needed for MolGANModel training."""


class MolGANLoss(
    namedtuple('MolGANLoss', tfgan.GANLoss._fields + (
       'valuenet_loss',
    ))):
    """Contains generator, discriminator, and value network losses."""


class MolGANTrainOps(namedtuple('GANTrainOps', tfgan.GANTrainOps._fields + (
    'valuenet_train_op',
))):
    """Contains training ops."""


def make_prediction_model(generator_inputs: tf.Tensor,
                          generator_fn,
                          value_fn) -> MolGANModel:
    with tf.variable_scope('Generator') as gen_scope:
        generated_data, predicted_data = generator_fn(generator_inputs)

    if value_fn is None:
        value_gen_outputs = None
    else:
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


def make_train_eval_model(real_data: Dict[str, tf.Tensor],
                          generator_inputs: tf.Tensor,
                          generator_fn,
                          discriminator_fn,
                          value_fn) -> Tuple[MolGANModel, Dict[str, tf.Tensor]]:
    with tf.variable_scope('Generator') as gen_scope:
        generated_data, predicted_data = generator_fn(generator_inputs)
    with tf.variable_scope('Discriminator') as dis_scope:
        discriminator_gen_outputs = discriminator_fn(generated_data)
    with tf.variable_scope(dis_scope, reuse=True):
        discriminator_real_outputs = discriminator_fn(real_data)

    for real_name, real_val in real_data.items():
        if real_name.startswith('reward'):
            continue
        gen_val = generated_data[real_name]
        if not gen_val.shape.is_compatible_with(real_val.shape):
            raise ValueError(
                'Generator output %s shape (%s) must be the same shape as real data '
                '(%s).' % (real_name, gen_val.shape, real_val.shape))

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
        generator_inputs,
        generated_data,
        generator_variables,
        gen_scope,
        generator_fn,
        real_data,
        discriminator_real_outputs,
        discriminator_gen_outputs,
        discriminator_variables,
        dis_scope,
        discriminator_fn,
        value_real_outputs,
        value_gen_outputs,
        value_variables,
        value_scope
    )
    return gan_model, predicted_data
