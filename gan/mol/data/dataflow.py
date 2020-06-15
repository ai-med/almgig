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
from itertools import chain
import logging
from typing import *
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tensorpack.dataflow import MapData, MapDataComponent, ProxyDataFlow, DataFlow

LOG = logging.getLogger(__name__)


class MapDataComponentAndAppend(ProxyDataFlow):
    """Append component to each data point.

    Parameters
    ----------
    ds : DataFlow
        Input DataFlow. ``dp[idx]`` has to be a list of :class:`nx.Graph`.
    idx : int
        Index of element to map.
    """
    def __init__(self, ds, func, index=0):
        super(MapDataComponentAndAppend, self).__init__(ds)

        self._index = index
        self._func = func

    def get_data(self):
        for dp in self.ds.get_data():
            newdp = list(dp)
            ret = self._func(dp[self._index])
            newdp.append(ret)
            yield newdp


class AppendNodeFeatures(ProxyDataFlow):
    """Append features of each node.

    Parameters
    ----------
    ds : DataFlow
        Input DataFlow. ``dp[idx]`` has to be a list of :class:`nx.Graph`.
    index : int
        Index of element holding graphs.
    data_key : str|list
        Name of node attribute holding node features.
    """
    def __init__(self, ds, index=0, data_key='label'):
        super(AppendNodeFeatures, self).__init__(ds)

        self._index = index
        self._data_key = data_key

    def _map_component(self, dp):
        graph = dp[self._index]
        assert isinstance(graph, nx.Graph), 'expected nx.Graph, but got {!r}'.format(graph)

        def get_feature(key):
            data = []
            for n, v in graph.nodes(data=key):
                assert v is not None, 'node property {} is None'.format(key)
                data.append(v)
            return data

        if isinstance(self._data_key, (tuple, list)):
            for dk in self._data_key:
                dp.append(get_feature(dk))
        else:
            dp.append(get_feature(self._data_key))

        return dp

    def get_data(self):
        for dp in self.ds.get_data():
            dp = list(dp)
            newdp = self._map_component(dp)
            yield newdp


class SparseOneHotEncoding(MapDataComponent):
    """One-hot encoding of features.

    Parameters
    ----------
    ds : DataFlow
        Input DataFlow. ``dp[idx]`` has to be a list of integer vectors.
    values : list
        A list of allowed values. The position determines the column
        used for one-hot encoding.
    index : int
        Index of element holding features.
    """

    def __init__(self, ds, values, index=0):
        MapDataComponent.__init__(self, ds, self._map_component, index)

        self._values = values

    def _map_component(self, features):
        n_nodes = sum(map(len, features))
        coords = np.empty((n_nodes, 2), dtype=np.int32)
        coords[:, 0] = np.arange(n_nodes, dtype=np.int32)
        shape = (n_nodes, len(self._values))
        for n, value in enumerate(chain(*features)):
            i = np.flatnonzero(value == self._values)
            coords[n, 1] = i
        values = np.ones(n_nodes, dtype=np.float32)
        return coords, values, shape


class DenseOneHotEncoding(MapDataComponent):
    """One-hot encoding of features.

    Parameters
    ----------
    ds : DataFlow
        Input DataFlow. ``dp[idx]`` has to be a list of integer vectors.
    values : array-like
        A list of allowed values. The position determines the column
        used for one-hot encoding.
    index : int
        Index of element holding features.
    """

    def __init__(self,
                 ds: DataFlow,
                 values: Union[np.ndarray, Sequence[Any]],
                 index: int = 0) -> None:
        MapDataComponent.__init__(self, ds, self._map_component, index)

        self._values = np.asanyarray(values)

    def _map_component(self, features: Union[List[int], np.ndarray]):
        features = np.asanyarray(features)

        n_nodes = features.shape[0]
        output = np.zeros((n_nodes, self._values.shape[0]), dtype=np.int32)
        for n, value in enumerate(features):
            i = np.flatnonzero(value == self._values)
            output[n, i] = 1

        return output


GraphValidatorFn = Callable[[nx.Graph], bool]


class GraphConvEmbedding(MapData):
    """Convert sparse matrices to dense and pad with zeros to
    obtain matrices of the same size.

    Resulting matrices have the same size, irrespective of the
    actual number of nodes in the graph. Thus, data can be batched.

    Input data point:
    0. Original nx.Graph
       -> Drop
    1. Sparse adjacency matrix to construct support from
       -> Convert to dense and pad
    2. Sparse adjacency matrix to use as label in BCE loss
       -> Convert to dense and pad
    3. Sparse indicator matrix to compute performance metrics from (only valid and test)
       -> Convert to dense and pad
    4. Node features
       -> Pad with zeros

    Parameters
    ----------
    ds : DataFlow
        Input data flow.
    max_length : int
        Number of nodes each adjacency matrix should have.
    validator : callable, optional
        Callable that checks whether a graph represents a valid molecule.
    """
    def __init__(self, ds: DataFlow,
                 max_length: int,
                 validator: Optional[GraphValidatorFn] = None) -> None:
        super(GraphConvEmbedding, self).__init__(ds, self._map_data)
        self._graph_index = 0
        self._adj_index = 1
        self._label_index = 2
        self._mask_index = 3

        if max_length <= 0:
            raise ValueError('max_length must be greater zero.')

        self.max_length = max_length
        self.validator = validator

    def _sparse_to_dense_padded(self, mat):
        shape = (self.max_length, self.max_length)
        if not isinstance(mat, sp.coo_matrix):
            mat = mat.tocoo()
        nmat = sp.coo_matrix((mat.data, (mat.row, mat.col)),
                             dtype=mat.dtype,
                             shape=shape)

        return nmat.toarray()

    def size(self):
        if self.validator:
            raise NotImplementedError()
        return super(GraphConvEmbedding, self).size()

    def _map_data(self, dp):
        newdp = []
        if self.validator is not None and not self.validator(dp[self._graph_index]):
            return None

        LOG.debug('Padding data to have %d nodes', self.max_length)
        adj = self._sparse_to_dense_padded(dp[self._adj_index])
        newdp.append(adj)

        label_adj = self._sparse_to_dense_padded(dp[self._label_index])
        newdp.append(label_adj)

        if dp[self._mask_index] is not None:
            label_mask = self._sparse_to_dense_padded(dp[self._mask_index])
            newdp.append(label_mask)

        for i in range(self._mask_index + 1, len(dp)):
            features = np.asanyarray(dp[i])
            if features.ndim == 1:
                features = features[:, np.newaxis]
            if features.shape[0] < self.max_length:
                pad_shape = list(features.shape)
                pad_shape[0] = self.max_length - features.shape[0]
                pad = np.zeros(pad_shape, dtype=features.dtype)
                values = np.concatenate((features, pad), axis=0)
            else:
                values = features
            newdp.append(values)

        # [0] = train adjacency
        # [1] = label adjacency
        # [2] = label adjacency mask indices (optional)
        # [3] = features
        return newdp


class AppendMolMetrics(ProxyDataFlow):
    """Append features of each node.

    Parameters
    ----------
    ds : DataFlow
        Input DataFlow. ``dp[idx]`` has to be a list of :class:`nx.Graph`.
    metrics_fn : callable
        A functions that takes a list of NumPy arrays pairs and returns a float.
        The first array is an edge encoding matrix, the
        second array a node encoding vector.
    index_edges : int
        Index of element holding edge encoding matrix.
    index_node : int
        Index of element holding node encoding vector.
    """
    def __init__(self, ds, metrics_fn, index_edges=0, index_node=1):
        super(AppendMolMetrics, self).__init__(ds)

        self._metrics_fn = metrics_fn
        self._index_edges = index_edges
        self._index_node = index_node

    def _map_component(self, dp):
        edges = dp[self._index_edges]
        nodes = dp[self._index_node]
        assert edges.ndim == 2 and nodes.ndim == 2

        score = self._metrics_fn([(nodes.squeeze(axis=1), edges)])
        assert np.isfinite(score).all(), '{} is not finite'.format(score)
        dp.append(score[0])

        return dp

    def get_data(self):
        for dp in self.ds.get_data():
            dp = list(dp)
            newdp = self._map_component(dp)
            yield newdp


class ColumnStack(MapData):
    def __init__(self, ds: DataFlow, indices: Sequence[int], dtype=np.float32) -> None:
        super(ColumnStack, self).__init__(ds, self._map_data)
        self.indices = indices
        self.dtype = dtype

    def _map_data(self, dp: List[np.ndarray]) -> List[np.ndarray]:
        arr = np.column_stack([dp[i] for i in self.indices])
        newdp = []
        for i, value in enumerate(dp):
            if i not in self.indices:
                newdp.append(value)
        newdp.append(arr)

        return newdp
