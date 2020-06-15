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

from ..data import create_dataflow
from ..data.graph2mol import MoleculeData

PredictInputFn = Callable[[], Dict[str, tf.Tensor]]


def set_batch_size(x, n):
    for v, t in x.items():
        s = t.get_shape().as_list()
        t.set_shape([n] + s[1:])
    return x


class BasePredictFunction:
    def __init__(self, num_latent: int) -> None:
        self.num_latent = num_latent

    def _get_noise_seed(self, batch_size: int, rnd: np.random.RandomState) -> np.ndarray:
        noise_seed = np.ones((batch_size, 1)) * rnd.normal(size=(1, self.num_latent))
        noise_seed = noise_seed.astype(np.float32)
        return noise_seed


class PredictionInputFunction(BasePredictFunction):
    def __init__(self,
                 data_dir: Path,
                 num_latent: int,
                 data_params: MoleculeData,
                 kind: str = 'valid') -> None:
        super(PredictionInputFunction, self).__init__(num_latent)
        self.data_dir = data_dir
        self.data_params = data_params
        self.kind = kind

    def _data_iter(self, n_samples: Optional[int], batch_size: int, seed: Optional[int]):
        rnd = np.random.RandomState(seed)
        noise_seed = self._get_noise_seed(batch_size, rnd)

        ds = create_dataflow(self.data_dir, self.kind, batch_size=batch_size)
        ds.reset_state()

        n = None
        if n_samples is not None:
            assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
            n = n_samples // batch_size
        for i, (adj_in, features, rewards) in enumerate(ds.get_data()):
            if i == n:
                break
            embedding = rnd.normal(size=(batch_size, self.num_latent))
            embedding = embedding.astype(np.float32)

            yield embedding, noise_seed, adj_in, features, rewards

    def create(self,
               n_samples: Optional[int] = 1000,
               batch_size: int = 1000,
               seed: Optional[int] = None) -> PredictInputFn:
        def _input_fn():
            max_nodes = self.data_params.MAX_NODES
            dataset = tf.data.Dataset.from_generator(
                lambda: self._data_iter(n_samples, batch_size, seed),
                (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32),
                (tf.TensorShape([batch_size, self.num_latent]),
                 tf.TensorShape([batch_size, self.num_latent]),
                 tf.TensorShape([batch_size, max_nodes, max_nodes]),
                 tf.TensorShape([batch_size, max_nodes]),
                 tf.TensorShape([batch_size, 1]))
            )

            iterator = dataset.make_one_shot_iterator()
            embedding, noise_seed, adj_in, features, rewards = iterator.get_next()

            next_input = {'embedding': embedding,
                          'noise': noise_seed,
                          'adjacency_in': adj_in,
                          'features': features,
                          'reward_real': rewards}
            for nam in ('adjacency_in', 'features'):
                tf.identity(next_input[nam], name=nam)
            return next_input

        return _input_fn


class GenerateInputFunction(BasePredictFunction):
    """Pass embedding vector, and generate graph from it."""

    def __init__(self, num_latent: int) -> None:
        super(GenerateInputFunction, self).__init__(num_latent)

    def create(self,
               embedding: np.ndarray,
               seed: Optional[int] = None) -> PredictInputFn:
        if embedding.ndim != 2:
            raise ValueError('expected 2D embedding vector, but has {} dimensions'.format(embedding.ndim))

        batch_size = embedding.shape[0]
        rnd = np.random.RandomState(seed)
        noise_seed = self._get_noise_seed(batch_size, rnd)

        def _input_fn():
            next_input = {'embedding': embedding,
                          'noise': noise_seed}
            ds = tf.data.Dataset.from_tensor_slices(next_input)
            ds = ds.batch(1)
            ds = ds.map(lambda x: set_batch_size(x, 1))

            next_input = ds.make_one_shot_iterator().get_next()

            return next_input

        return _input_fn

    @property
    def predict_keys(self) -> Tuple[str, str]:
        return 'features', 'adjacency'


class EmbedAndReconstructInputFunction(BasePredictFunction):
    """Pass graph and generate embedding vector and reconstructed graph from it."""

    def __init__(self, num_latent: int) -> None:
        super(EmbedAndReconstructInputFunction, self).__init__(num_latent=num_latent)

    def create(self,
               nodes: np.ndarray,
               edges: np.ndarray,
               seed: Optional[int] = None) -> PredictInputFn:
        if nodes.ndim != 2:
            raise ValueError(
                'expected 2D nodes array, but has {} dimensions'.format(
                    nodes.ndim))
        if edges.ndim != 3:
            raise ValueError(
                'expected 3D adjacency array, but has {} dimensions'.format(
                    edges.ndim))

        batch_size = nodes.shape[0]
        if batch_size != edges.shape[0]:
            raise ValueError('batch dimension must be equal: '
                             '{} != {}'.format(batch_size, edges.shape[0]))

        rnd = np.random.RandomState(seed)
        noise = self._get_noise_seed(batch_size, rnd)

        def predict_fn():
            next_input = {'noise': noise,
                          'adjacency_in': edges,
                          'features': nodes}
            ds = tf.data.Dataset.from_tensor_slices(next_input)
            ds = ds.batch(1)
            ds = ds.map(lambda x: set_batch_size(x, 1))

            next_input = ds.make_one_shot_iterator().get_next()

            for nam in ('adjacency_in', 'features'):
                tf.identity(next_input[nam], name=nam)

            return next_input

        return predict_fn

    @property
    def predict_keys(self) -> Tuple[str, str, str]:
        return 'embedding', 'reconstructed/features', 'reconstructed/adjacency'


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
            noise=tf.placeholder(tf.float32, [self.batch_size, self.num_latent], 'noise'),
            valence=tf.placeholder(tf.float32, [self.data_params.NUM_NODE_TYPES + 1], 'valence'),
        )
        return placeholders

    def __call__(self) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        # initialize dataflow
        ds = self._ds
        ds.reset_state()
        self._data_iter = ds.get_data()
        self._data = {}

        with tf.name_scope("load_data"):
            placeholders = self._create_placeholders()

        self._placeholders = placeholders
        self._rnd = np.random.RandomState()

        next_batch = placeholders.copy()
        latent = next_batch.pop('noise')
        return latent, next_batch

    def _sample_latent(self):
        z = self._rnd.normal(size=(self.batch_size, self.num_latent))
        return z.astype(np.float32)

    def get_next_batch(self) -> None:
        adj_in, features, rewards = next(self._data_iter)
        z = self._sample_latent()
        noise = self._sample_latent()

        self._data = {
            'embedding': z,
            'adjacency_in': adj_in,
            'features': features,
            'reward_real': rewards,
            'noise': noise,
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
