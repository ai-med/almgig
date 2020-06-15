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
import tensorflow.contrib.gan as tfgan

from .base import MolGANLoss, MolGANModel
from .losses import non_saturating_loss, reward_loss
from .molgan import MolGANEstimator, _split_inputs
from .regularization import gradient_penalty, variance_penalty


class NonSaturatingMolGANEstimator(MolGANEstimator):

    def _make_loss(self,
                   model: MolGANModel,
                   mode: str) -> MolGANLoss:
        gan_loss = non_saturating_loss(
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
