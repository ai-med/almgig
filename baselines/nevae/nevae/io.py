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
import pickle
from typing import List, Optional, Tuple
import networkx as nx
import numpy as np
import tensorflow as tf

LOG = logging.getLogger(__name__)


GraphData = namedtuple('GraphData', (
    'adjacency',
    'adjacency_lower',
    'edge_weights',
    'features',
    'log_weight_indicator',
    'edge_index',
    'edge_list',
    'num_edges',
))


class PredictInputFunction:
    def __init__(self,
                 n_samples: int,
                 n_nodes: int,
                 n_latent: int,
                 seed: Optional[int] = None) -> None:
        self._samples = n_samples
        self._nodes = n_nodes
        self._latent = n_latent
        self._random_state = np.random.RandomState(seed)

    def __call__(self):
        def _sample_iter():
            for _ in range(self._samples):
                noise = self._random_state.randn(self._nodes, self._latent)
                yield noise.astype(np.float32)

        dataset = tf.data.Dataset.from_generator(
            _sample_iter,
            tf.float32,
            tf.TensorShape([self._nodes, self._latent])
        )

        iterator = dataset.make_one_shot_iterator()
        noise_tensor = iterator.get_next()

        return {'noise': noise_tensor}


class GraphTransformer:
    def __init__(self,
                 n_nodes: int,
                 n_node_types: int,
                 n_edge_types: int,
                 max_edges: int) -> None:
        self._nodes = n_nodes
        self._node_types = n_node_types
        self._edge_types = n_edge_types

        # number of elements in the lower triangle, including main diagonal
        self._n_adj_flat = n_nodes * (n_nodes + 1) // 2
        self._max_edges = max_edges
        # C, F, H, N, O
        self._valence = np.array([4, 1, 1, 3, 2], dtype=np.float32)

    @property
    def num_nodes(self):
        return self._nodes

    def _get_transformed_inputs(self,
                                features: np.ndarray,
                                edges: np.ndarray,
                                invalid_prob: float = 1e-9) -> GraphData:
        assert edges.shape[0] == edges.shape[1], '{} != {}'.format(edges.shape[0], edges.shape[1])
        assert edges.shape[0] == self._nodes, '{} != {}'.format(edges.shape[0], self._nodes)
        assert features.shape[0] == self._nodes, '{} != {}'.format(features.shape[0], self._nodes)
        assert edges.max() <= self._edge_types, '{} > {}'.format(edges.max(), self._edge_types)
        assert edges.min() == 0, '{} != 0'.format(edges.min())

        nn = self._nodes

        # lower triangle
        idx_lower = np.tril_indices(nn)
        row, col = idx_lower
        assert row.shape[0] == self._n_adj_flat

        # degree[i]: degree of nodes before adding i-th edge
        degree = []
        for i, (x, y) in enumerate(zip(row, col)):
            weight = edges[x, y]
            if weight == 0:
                continue
            if len(degree) == 0:
                nc = np.zeros(nn)
            else:
                nc = degree[-1].copy()
            nc[x] += weight
            nc[y] += weight
            degree.append(nc)

        # weight_indicators[i]: binary mask of allowed edge types before
        # adding i-th edge
        vv = self._valence[features.argmax(axis=-1)]
        indicator = np.ones((nn, self._edge_types), dtype=np.float32)
        for i, left in enumerate(vv.astype(int)):
            indicator[i, left:] = 0
        weight_indicators = [indicator]

        for deg in degree[:-1]:  # we don't need valence _after_ adding the last edge
            indicator = weight_indicators[-1].copy()

            d = (vv - deg).astype(int)
            for i, left in enumerate(d):
                indicator[i, left:] = 0
            weight_indicators.append(indicator)

        num_edges = len(weight_indicators)
        assert np.count_nonzero(edges[idx_lower]) == num_edges
        # shape = (num_edges, num_nodes, num_edge_types)
        weight_indicators = np.asarray(weight_indicators, dtype=np.float32)

        # zero-padding if num_edges < max_edges
        null_edges = self._max_edges - num_edges
        if null_edges > 0:
            null_ind = np.zeros((null_edges, edges.shape[0], self._edge_types), dtype=np.float32)
            weight_indicators = np.row_stack((weight_indicators, null_ind))

        # mask zero values
        weight_indicators = np.log((1.0 - weight_indicators) * invalid_prob + weight_indicators)

        adj_one_hot = np.zeros((self._edge_types, nn, nn), dtype=np.int32)
        # lower triangle of one-hot encoded edges matrix
        weight_bin = np.zeros([self._n_adj_flat, self._edge_types], dtype=np.int32)
        # indices of true edges (wrt flat lower triangular matrix)
        coord = np.zeros(self._max_edges, dtype=np.int32)
        # edge list of true edges (with row, col index)
        edge_list = np.zeros((self._max_edges, 2), dtype=np.int32)
        cur_edge = 0
        for k, (i, j) in enumerate(zip(row, col)):
            if edges[i, j] > 0:
                ti = edges[i, j] - 1

                adj_one_hot[ti, i, j] = 1
                adj_one_hot[ti, j, i] = 1

                weight_bin[k, ti] = 1
                coord[cur_edge] = k
                edge_list[cur_edge, :] = i, j
                cur_edge += 1
        assert cur_edge == num_edges

        adj = np.zeros(edges.shape, dtype=np.int32)
        adj[edges.nonzero()] = 1
        adj_lower = adj[idx_lower][:, np.newaxis]

        return GraphData(adj_one_hot, adj_lower, weight_bin, features,
                         weight_indicators,
                         coord, edge_list, num_edges)

    def _get_edges_and_features(self,
                                graph: nx.Graph,
                                source_node: int) -> Tuple[np.ndarray, np.ndarray]:
        assert source_node < self._nodes

        node_order = []
        node_set = set()
        for edge in nx.bfs_edges(graph, source_node):
            for node in edge:
                if node not in node_set:
                    node_order.append(node)
                    node_set.add(node)
        assert len(node_order) == graph.number_of_nodes()

        features = np.zeros((self._nodes, self._node_types), dtype=np.int32)
        for ui, u in enumerate(node_order):
            idx = graph.nodes[u]['AtomCode'] - 1
            features[ui, idx] = 1

        weight = nx.adjacency_matrix(
            graph, nodelist=node_order).toarray()
        return weight, features

    def _graph_to_arrays(self, graph: nx.Graph, source_node: int) -> GraphData:
        weight, features = self._get_edges_and_features(graph, source_node)

        return self._get_transformed_inputs(features, weight)

    def convert(self, graph: nx.Graph) -> GraphData:
        source_node = np.random.randint(self._nodes)
        return self._graph_to_arrays(graph, source_node)

    @property
    def output_types(self):
        return tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int32, tf.int64

    @property
    def output_shapes(self):
        nn = self._nodes
        return (tf.TensorShape([self._edge_types, nn, nn]),  # adj
                tf.TensorShape([self._n_adj_flat, 1]),  # tril adj
                tf.TensorShape([self._n_adj_flat, self._edge_types]),  # tril edge types
                tf.TensorShape([nn, self._node_types]),  # features
                tf.TensorShape([self._max_edges, nn, self._edge_types]),  # log(weight_indicators)
                tf.TensorShape([self._max_edges]),  # coordinates
                tf.TensorShape([self._max_edges, 2]),  # edge list
                tf.TensorShape([]))


class SequenceGraphTransformer(GraphTransformer):
    def __init__(self,
                 n_nodes: int,
                 n_node_types: int,
                 n_edge_types: int) -> None:
        super().__init__(n_nodes, n_node_types, n_edge_types, None)

    def _get_transformed_inputs(self,
                                features: np.ndarray,
                                edges: np.ndarray,
                                invalid_prob: float = 1e-9) -> GraphData:
        assert edges.shape[0] == edges.shape[1], '{} != {}'.format(edges.shape[0], edges.shape[1])
        # assert edges.shape[0] == self._nodes, '{} != {}'.format(edges.shape[0], self._nodes)
        # assert features.shape[0] == self._nodes, '{} != {}'.format(features.shape[0], self._nodes)
        assert edges.max() <= self._edge_types, '{} > {}'.format(edges.max(), self._edge_types)
        assert edges.min() == 0, '{} != 0'.format(edges.min())

        nn = features.shape[0]

        # lower triangle
        idx_lower = np.tril_indices(nn)
        row, col = idx_lower
        """
        assert row.shape[0] == self._n_adj_flat
        # degree[i]: degree of nodes before adding i-th edge
        degree = []
        for i, (x, y) in enumerate(zip(row, col)):
            weight = edges[x, y]
            if weight == 0:
                continue
            if len(degree) == 0:
                nc = np.zeros(nn)
            else:
                nc = degree[-1].copy()
            nc[x] += weight
            nc[y] += weight
            degree.append(nc)

        # weight_indicators[i]: binary mask of allowed edge types before
        # adding i-th edge
        vv = self._valence[features.argmax(axis=-1)]
        indicator = np.ones((nn, self._edge_types), dtype=np.float32)
        for i, left in enumerate(vv.astype(int)):
            indicator[i, left:] = 0
        weight_indicators = [indicator]

        for deg in degree[:-1]:  # we don't need valence _after_ adding the last edge
            indicator = weight_indicators[-1].copy()

            d = (vv - deg).astype(int)
            for i, left in enumerate(d):
                indicator[i, left:] = 0
            weight_indicators.append(indicator)

        num_edges = len(weight_indicators)
        assert np.count_nonzero(edges[idx_lower]) == num_edges
        # shape = (num_edges, num_nodes, num_edge_types)
        weight_indicators = np.asarray(weight_indicators, dtype=np.float32)

        # zero-padding if num_edges < max_edges
        null_edges = self._max_edges - num_edges
        if null_edges > 0:
            null_ind = np.zeros((null_edges, edges.shape[0], self._edge_types), dtype=np.float32)
            weight_indicators = np.row_stack((weight_indicators, null_ind))

        # mask zero values
        weight_indicators = np.log((1.0 - weight_indicators) * invalid_prob + weight_indicators)
        """

        adj_incr = [np.zeros((self._edge_types, nn, nn), dtype=np.int32)]
        """
        # lower triangle of one-hot encoded edges matrix
        weight_bin = np.zeros([self._n_adj_flat, self._edge_types], dtype=np.int32)
        # indices of true edges (wrt flat lower triangular matrix)
        coord = np.zeros(self._max_edges, dtype=np.int32)
        # edge list of true edges (with row, col index)
        edge_list = np.zeros((self._max_edges, 2), dtype=np.int32)
        cur_edge = 0
        """
        for k, (i, j) in enumerate(zip(row, col)):
            if edges[i, j] > 0:
                ti = edges[i, j] - 1

                adj = adj_incr[-1].copy()
                adj[ti, i, j] = 1
                adj[ti, j, i] = 1
                adj_incr.append(adj)

        adj = adj_incr[-1]
        adj_incr = np.stack(adj_incr[1:], axis=0)
        num_edges = adj_incr.shape[0]

        return adj, adj_incr, features, num_edges

    @property
    def output_types(self):
        return tf.int32, tf.int32, tf.int32, tf.int32

    @property
    def output_shapes(self):
        return (tf.TensorShape([self._edge_types, None, None]),  # adj
                tf.TensorShape([None, self._edge_types, None, None]),  # incremental adj
                tf.TensorShape([None, self._node_types]),  # features
                tf.TensorShape([]))  # num edges


class GraphLoader:

    def __init__(self, filename: str, num_nodes: int) -> None:
        self._filename = filename
        self._nodes = num_nodes
        self._max_edges = None

    @property
    def num_nodes(self) -> int:
        return self._nodes

    @property
    def max_edges(self) -> Optional[int]:
        return self._max_edges

    def _load_data(self):
        with open(self._filename, 'rb') as fp:
            data = pickle.load(fp)
        return data

    def get_next(self):
        nn = self._nodes
        max_edges = 0
        graphs = self._load_data()
        for G_org in filter(lambda g: g.number_of_nodes() == nn, graphs):
            G = nx.Graph()
            for no, d in G_org.nodes(data=True):
                G.add_node(no - 1, **d)
            for u, v, d in G_org.edges(data="BondTypeCode"):
                G.add_edge(u - 1, v - 1, weight=d)

            assert G.number_of_nodes() == nn, "{} != {}".format(
                G.number_of_nodes(), nn)
            max_edges = max(max_edges, G.number_of_edges())
            yield G

        self._max_edges = max_edges

    def get_all(self) -> List[nx.Graph]:
        return list(self.get_next())


class InputFunction:
    def __init__(self,
                 graphs: List[nx.Graph],
                 graph_transformer: GraphTransformer,
                 epochs: int,
                 n_latent: int,
                 shuffle: bool = False) -> None:
        self._graphs = graphs
        self._graph_transformer = graph_transformer
        self._epochs = epochs
        self._latent = n_latent
        self._shuffle = shuffle

    def __call__(self):
        def _data_iter():
            idx = np.arange(len(self._graphs), dtype=int)
            for _ in range(self._epochs):
                if self._shuffle:
                    np.random.shuffle(idx)
                for i in idx:
                    d = self._graph_transformer.convert(self._graphs[i])
                    yield d

        dataset = tf.data.Dataset.from_generator(
            _data_iter,
            self._graph_transformer.output_types,
            self._graph_transformer.output_shapes)
        dataset = dataset.prefetch(128)
        iterator = dataset.make_one_shot_iterator()

        nn = self._graph_transformer.num_nodes
        noise = tf.random_normal(shape=[nn, self._latent], stddev=1.0)
        data = iterator.get_next()

        datad = {k: v for k, v in zip(GraphData._fields, data)}
        return datad, noise


class DumpWeightsHook(tf.train.SessionRunHook):

    def __init__(self):
        self._weights = None

    @property
    def weights(self):
        return self._weights

    def after_create_session(self, session, coord):
        weights = tf.trainable_variables()

        values = session.run(weights)
        outputs = {}
        for weight, value in zip(weights, values):
            outputs[weight.name] = value
        self._weights = outputs
