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


class GANSummarySaverHook(tf.train.SummarySaverHook):

    DEFAULT_SCOPES = (
        'Discriminator',
        'Generator',
        'discriminator_train',
        'generator_train',
        'DiscriminatorLoss',
        'GeneratorLoss',
    )

    def __init__(self,
                 save_steps=None,
                 save_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 scopes=None):
        super(GANSummarySaverHook, self).__init__(
            save_steps=save_steps,
            save_secs=save_secs,
            output_dir=output_dir,
            summary_writer=summary_writer,
            summary_op='',
        )
        self._scopes = scopes

    def _get_summary_op(self):
        if self._scopes is None:
            scopes = GANSummarySaverHook.DEFAULT_SCOPES
        else:
            scopes = self._scopes
            if not isinstance(scopes, list):
                scopes = [scopes]

        g = tf.get_default_graph()
        v = []
        for scope in scopes:
            v.extend(g.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
        return v


class CollectTensorHook(tf.train.SessionRunHook):

    def __init__(self, name):
        self._name = name
        self._tensor = None
        self._data = None

    def after_create_session(self, session, coord):
        g = tf.get_default_graph()
        self._tensor = g.get_tensor_by_name(self._name)
        self._data = []

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=self._tensor)

    def after_run(self, run_context, run_values):
        self._data.append(run_values.results)

    @property
    def data(self):
        return self._data
