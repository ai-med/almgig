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
from typing import Callable, List
import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from .base import MolGANLoss, MolGANModel, MolGANTrainOps

EstimatorTrainHooks = List[tf.train.SessionRunHook]
MolGANTrainHooksFn = Callable[[MolGANTrainOps, tfgan.GANTrainSteps], EstimatorTrainHooks]


def make_train_estimator_spec(gan_model: MolGANModel,
                              gan_loss: MolGANLoss,
                              hooks_fn: MolGANTrainHooksFn,
                              generator_learning_rate: float,
                              discriminator_learning_rate: float,
                              beta1: float = 0.9,
                              beta2: float = 0.999,
                              add_summaries: bool = False,
                              summarize_gradients: bool = False) -> tf.estimator.EstimatorSpec:
    gen_lr = tf.Variable(generator_learning_rate, name='generator_learning_rate',
                         trainable=False, dtype=tf.float32)
    dis_lr = tf.Variable(discriminator_learning_rate, name='discriminator_learning_rate',
                         trainable=False, dtype=tf.float32)
    if add_summaries:
        tf.summary.scalar('generator_learning_rate', gen_lr)
        tf.summary.scalar('discriminator_learning_rate', dis_lr)

    generator_optimizer = tf.train.AdamOptimizer(gen_lr, beta1, beta2)
    discriminator_optimizer = tf.train.AdamOptimizer(dis_lr, beta1, beta2)
    train_ops = tfgan.gan_train_ops(
        gan_model, gan_loss,
        generator_optimizer, discriminator_optimizer,
        summarize_gradients=summarize_gradients)
    ops_kwargs = train_ops._asdict()

    scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss

    if gan_loss.valuenet_loss is not None:
        valuenet_step = tf.get_variable(
            'dummy_global_step_valuenet',
            shape=[],
            dtype=tf.int64,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        valuenet_optimizer = tf.train.AdamOptimizer(learning_rate)

        valuenet_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, gan_model.valuenet_scope.name)
        with tf.name_scope('valuenet_train'):
            valuenet_train_op = tf.contrib.training.create_train_op(
                total_loss=gan_loss.valuenet_loss,
                optimizer=valuenet_optimizer,
                variables_to_train=gan_model.valuenet_variables,
                global_step=valuenet_step,
                update_ops=valuenet_update_ops,
                summarize_gradients=summarize_gradients)

        ops_kwargs['valuenet_train_op'] = valuenet_train_op
    else:
        ops_kwargs['valuenet_train_op'] = None

    train_steps = tfgan.GANTrainSteps(1, 1)
    train_ops = MolGANTrainOps(**ops_kwargs)
    training_hooks = hooks_fn(train_ops, train_steps)

    return tf.estimator.EstimatorSpec(
        loss=scalar_loss,
        mode=tf.estimator.ModeKeys.TRAIN,
        train_op=train_ops.global_step_inc_op,
        training_hooks=training_hooks)


def make_eval_estimator_spec(gan_model: MolGANModel,
                             gan_loss: MolGANLoss) -> tf.estimator.EstimatorSpec:
    scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
    with tf.name_scope(None, 'metrics',
                       [gan_loss.generator_loss,
                        gan_loss.discriminator_loss]):
        eval_metric_ops = {
            'generator_loss':
                tf.metrics.mean(gan_loss.generator_loss),
            'discriminator_loss':
                tf.metrics.mean(gan_loss.discriminator_loss)
        }
        if gan_loss.valuenet_loss is not None:
            eval_metric_ops['valuenet_loss'] = tf.metrics.mean(
                gan_loss.valuenet_loss)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        predictions=gan_model.generated_data,
        loss=scalar_loss,
        eval_metric_ops=eval_metric_ops)
