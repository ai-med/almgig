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
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as td
from tensorpack.utils.serialize import loads_msgpack

from .graph2mol import MoleculeData


def _decode_data(dp):
    return loads_msgpack(dp[1])


def _squeeze_last(x):
    return np.squeeze(x, axis=-1)


def create_dataflow(data_dir: Path,
                    kind: str,
                    batch_size: int,
                    shuffle: bool = True) -> td.DataFlow:

    path = data_dir / "{}.mdb".format(kind)
    ds = td.LMDBData(str(path), shuffle=shuffle)
    ds = td.MapData(ds, _decode_data)
    ds = td.BatchData(ds, batch_size, remainder=False)
    ds = td.MapDataComponent(ds, _squeeze_last, index=1)
    return ds


def make_predict_fn(data_dir: Path,
                    num_latent: int,
                    n_samples: int = 1000,
                    batch_size: int = 1000,
                    seed: Optional[int] = None) -> Callable[[], tf.Tensor]:
    assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
    def _input_fn():
        rnd = np.random.RandomState(seed)
        inputs = rnd.normal(size=(n_samples, num_latent))
        inputs = inputs.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(batch_size)
        def _set_shape(x):
            x.set_shape([batch_size, num_latent])
            return x
        dataset = dataset.map(_set_shape)

        iterator = dataset.make_one_shot_iterator()
        next_input = iterator.get_next()
        return next_input
    return _input_fn


class InputFunction:

    def __init__(self,
                 data_flow: td.DataFlow,
                 batch_size: int,
                 num_latent: int,
                 data_params: MoleculeData) -> None:
        self._ds = data_flow
        self.batch_size = batch_size
        self.num_latent = num_latent
        self.data_params = data_params
        self._iterator = None
        self._placeholders = None
        self._rnd = None

        self._valence = data_params.valence_of_supported_atoms

    def _create_placeholders(self):
        max_nodes = self.data_params.MAX_NODES
        adj_shape = [None, max_nodes, max_nodes]
        feat_shape = [None, max_nodes]
        reward_shape = [None, 1]

        placeholders = dict(
            embedding=tf.placeholder(tf.float32, [self.batch_size, self.num_latent], 'embedding_in'),
            adjacency_in=tf.placeholder(tf.int32, adj_shape, 'adjacency_in'),
            features=tf.placeholder(tf.int32, feat_shape, 'features_in'),
            reward_real=tf.placeholder(tf.float32, reward_shape, 'reward_real'),
            reward_generated=tf.placeholder(tf.float32, reward_shape, 'reward_generated'),
            valence=tf.placeholder(tf.float32, [self.data_params.NUM_NODE_TYPES + 1], 'valence'),
        )
        return placeholders

    def __call__(self) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        # initialize dataflow
        self._ds.reset_state()
        self._data_iter = self._ds.get_data()
        self._data = {}

        with tf.name_scope("load_data"):
            placeholders = self._create_placeholders()

        self._placeholders = placeholders
        self._rnd = np.random.RandomState()

        next_batch = placeholders.copy()
        latent = next_batch.pop('embedding')
        return latent, next_batch

    def _sample_latent(self):
        z = self._rnd.normal(size=(self.batch_size, self.num_latent))
        return z.astype(np.float32)

    def get_next_batch(self) -> None:
        adj_in, features, rewards = next(self._data_iter)
        z = self._sample_latent()

        self._data = {
            'embedding': z,
            'adjacency_in': adj_in,
            'features': features,
            'reward_real': rewards,
            'valence': self._valence,
        }

    @property
    def data(self) -> Dict[str, tf.Tensor]:
        return self._data

    @property
    def feed_dict(self) -> Dict[tf.Tensor, np.ndarray]:
        new_feed_dict = {}
        for k, v in self._data.items():
            new_feed_dict[self._placeholders[k]] = v
        return new_feed_dict
