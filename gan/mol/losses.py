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
import tensorflow.contrib.gan as tfgan

from .base import MolGANLoss, MolGANModel


def wasserstein_loss(model: tfgan.GANModel,
                     add_summaries: bool = False,
                     scope: Optional[str] = None) -> tfgan.GANLoss:
    with tf.name_scope(scope, 'WassersteinLoss',
                       [model.discriminator_real_outputs, model.discriminator_gen_outputs]):
        loss_D = wasserstein_discriminator_loss(model, add_summaries=add_summaries)
        loss_G = wasserstein_generator_loss(model, add_summaries=add_summaries)

    gan_loss = tfgan.GANLoss(generator_loss=loss_G, discriminator_loss=loss_D)

    return gan_loss


def wasserstein_discriminator_loss(model: tfgan.GANModel,
                                   add_summaries: bool = False,
                                   scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'DiscriminatorLoss',
                       values=[model.discriminator_real_outputs, model.discriminator_gen_outputs]):
        loss_on_generated = tf.reduce_mean(model.discriminator_gen_outputs)
        loss_on_real = tf.reduce_mean(model.discriminator_real_outputs)
        loss_D = loss_on_generated - loss_on_real
        tf.losses.add_loss(loss_D)

        if add_summaries:
            tf.summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
            tf.summary.scalar('discriminator_real_wass_loss', loss_on_real)
            tf.summary.scalar('discriminator_wass_loss', loss_D)
        return loss_D


def wasserstein_generator_loss(model: tfgan.GANModel,
                               add_summaries: bool = False,
                               scope: Optional[str] = None) -> tf.Tensor:
    with tf.name_scope(scope, 'GeneratorLoss',
                       [model.discriminator_gen_outputs]):
        loss_on_generated = tf.reduce_mean(model.discriminator_gen_outputs)
        loss_G = -loss_on_generated
        tf.losses.add_loss(loss_G)
        if add_summaries:
            tf.summary.scalar('generator_wass_loss', loss_G)

    return loss_G


def non_saturating_loss(model: tfgan.GANModel,
                        add_summaries: bool = False,
                        scope: Optional[str] = None) -> tfgan.GANLoss:
    """Non-saturating Generative Adverserial Networks.

     The loss for the generator is computed using the log trick. That is,
     `G_loss = -log(D(fake_images))  [maximizes log(D)]`.
    """
    with tf.name_scope(scope, 'GANLoss',
                       [model.discriminator_real_outputs, model.discriminator_gen_outputs]):
        loss_D = non_saturating_discriminator_loss(model, add_summaries=add_summaries)
        loss_G = non_saturating_generator_loss(model, add_summaries=add_summaries)

    gan_loss = tfgan.GANLoss(generator_loss=loss_G, discriminator_loss=loss_D)

    return gan_loss


def non_saturating_discriminator_loss(model: tfgan.GANModel,
                                      add_summaries: bool = False,
                                      scope: Optional[str] = None) -> tfgan.GANLoss:
    with tf.name_scope(scope, 'DiscriminatorLoss',
                       [model.discriminator_real_outputs, model.discriminator_gen_outputs]):
        loss_on_generated = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.discriminator_gen_outputs,
                labels=tf.zeros_like(model.discriminator_gen_outputs)))
        loss_on_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.discriminator_real_outputs,
                labels=tf.ones_like(model.discriminator_real_outputs)))

        # Total discriminator loss.
        loss_D = loss_on_real + loss_on_generated
        tf.losses.add_loss(loss_D)

        if add_summaries:
            tf.summary.scalar('discriminator_gen_gan_loss', loss_on_generated)
            tf.summary.scalar('discriminator_real_gan_loss', loss_on_real)
            tf.summary.scalar('discriminator_gan_loss', loss_D)
    return loss_D


def non_saturating_generator_loss(model: tfgan.GANModel,
                                  add_summaries: bool = False,
                                  scope: Optional[str] = None) -> tfgan.GANLoss:
    with tf.name_scope(scope, 'GeneratorLoss',
                       [model.discriminator_gen_outputs]):
        loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.discriminator_gen_outputs,
                labels=tf.ones_like(model.discriminator_gen_outputs)))
        tf.losses.add_loss(loss_G)
        if add_summaries:
            tf.summary.scalar('generator_gan_loss', loss_G)
    return loss_G


def reward_loss(model: MolGANModel,
                reward_real: tf.Tensor,
                reward_generated: tf.Tensor,
                gan_loss: tfgan.GANLoss,
                lam: float,
                add_summaries: bool = False,
                scope: Optional[str] = None) -> MolGANLoss:
    with tf.name_scope(scope, 'ValueLoss'):
        loss_on_real = tf.losses.mean_squared_error(
            reward_real, model.valuenet_real_probs)
        loss_on_generated = tf.losses.mean_squared_error(
            reward_generated, model.valuenet_gen_probs)
        loss_V = 0.5 * (loss_on_real + loss_on_generated)
        tf.losses.add_loss(loss_V)

        # maximize reward on generated
        gen_reward = tf.reduce_mean(model.valuenet_gen_probs)
        gen_value_loss = -gen_reward
        gen_loss = gan_loss.generator_loss

        alpha = tf.abs(tf.stop_gradient(gen_loss / gen_value_loss))
        lam = tf.assert_scalar(lam)

        generator_loss = lam * gen_loss + (1.0 - lam) * alpha * gen_value_loss

        if add_summaries:
            tf.summary.scalar('lambda', lam)
            tf.summary.scalar('alpha', alpha)
            tf.summary.scalar('reward_on_generated', gen_reward)
            tf.summary.scalar('valuenet_gen_mse_loss', loss_on_generated)
            tf.summary.scalar('valuenet_real_mse_loss', loss_on_real)
            tf.summary.scalar('valuenet_mse_loss', loss_V)

    return MolGANLoss(generator_loss=generator_loss,
                      discriminator_loss=gan_loss.discriminator_loss,
                      valuenet_loss=loss_V)
