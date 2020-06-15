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
import argparse
import logging
from pathlib import Path
import pickle
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
from tensorpack.dataflow import DataFromList, SelectComponent
from tensorpack.dataflow.serialize import LMDBSerializer

from gan.mol.data.dataflow import AppendNodeFeatures, AppendMolMetrics, GraphConvEmbedding
from gan.mol.data.graph2mol import get_dataset, get_decoder
from gan.mol.metrics import GraphMolecularMetrics, RewardType
from gan.mol.metrics.scores import ValidityScore

LOG = logging.getLogger(__name__)

GraphValidatorFn = Callable[[nx.Graph], bool]
GraphAdjacencyTuple = Tuple[nx.Graph, sp.spmatrix, sp.spmatrix, Optional[sp.spmatrix]]


def sparse_remove_diagonal_elements(adj):
    assert isinstance(adj, sp.spmatrix)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    return adj


class Graph2MolValidator:

    def __init__(self) -> None:
        self._n_errors = 0

    @property
    def n_errors(self) -> int:
        return self._n_errors

    def __call__(self, graph: nx.Graph) -> bool:
        from rdkit import Chem
        from gan.mol.data import GDB9

        mol = Chem.RWMol()
        for n, d in graph.nodes(data=True):
            mol.AddAtom(Chem.Atom(d['Symbol']))
        offset = min(graph.nodes)

        bond_decoder = GDB9().get_bond_decoder()
        for start, end, d in graph.edges(data=True):
            start = int(start) - offset
            end = int(end) - offset
            mol.AddBond(start, end, bond_decoder[d['BondType']])

        success = Chem.SanitizeMol(mol, catchErrors=True)
        ret = success == 0
        if not ret:
            self._n_errors += 1

        if ValidityScore.to_valid_smiles(mol) is None:
            raise ValueError('{} represents no valid SMILES'.format(graph.graph['name']))
        return ret


def graph_to_adj(graph: nx.Graph) -> GraphAdjacencyTuple:
    adj_orig = nx.to_scipy_sparse_matrix(graph, weight='BondTypeCode', dtype=np.int32)
    adj_nodiag = sparse_remove_diagonal_elements(adj_orig)
    return graph, adj_orig, adj_nodiag, None


def read_graph_data(filename: Path, desired_max_nodes: int) -> Tuple[List[GraphAdjacencyTuple], int]:
    filename = Path(filename)
    LOG.info('Loading data from %s', filename)
    with filename.open('rb') as fp:
        data = pickle.load(fp)

    data = [g for g in data if g.number_of_nodes() <= desired_max_nodes]

    max_nodes = max((g.number_of_nodes() for g in data))
    LOG.info('Loaded %d graphs with at most %d nodes', len(data), max_nodes)

    unique_symbols = {d[1] for g in data for d in g.nodes(data='Symbol')}
    unique_symbols = np.array(list(unique_symbols))
    unique_symbols.sort()
    LOG.info('Data has %d unique node types: %r', len(unique_symbols), unique_symbols)

    return [graph_to_adj(d) for d in data], max_nodes


def create_dataflow(graphs: List[GraphAdjacencyTuple],
                    max_nodes: int,
                    metrics_fn: Callable[[np.ndarray, np.ndarray], float],
                    validator: Optional[GraphValidatorFn] = None,
                    shuffle: bool = False) -> SelectComponent:
    ds = DataFromList(graphs, shuffle)
    ds = AppendNodeFeatures(ds, data_key='AtomCode')
    ds_conv = GraphConvEmbedding(ds, max_nodes, validator)
    ds = AppendMolMetrics(ds_conv, metrics_fn, index_edges=0, index_node=2)
    ds = SelectComponent(ds, [0, 2, 3])

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graphs', metavar='PICKLE_FILE',
                        help='Path to pickled graphs.')
    parser.add_argument('--data', choices=['gdb9', 'zinc'], required=True,
                        help='Data to serialize.')
    parser.add_argument('--reward_type', type=RewardType.from_string,
                        metavar=RewardType.metavar(),
                        required=True)
    parser.add_argument('--norm_file',
                        help='Path to file to standardize penalized logP score.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    filename = Path(args.graphs)
    data, max_nodes = read_graph_data(filename, get_dataset(args.data).MAX_NODES)

    validator = Graph2MolValidator()

    conv = get_decoder(args.data)
    mol_metrics = GraphMolecularMetrics(conv, args.reward_type, args.norm_file)

    outfile = filename.with_suffix('.mdb')
    LOG.info("Saving to %s", outfile)
    ds = create_dataflow(data,
                         max_nodes=max_nodes,
                         metrics_fn=mol_metrics.get_reward_metrics,
                         validator=validator,
                         shuffle=False)
    LMDBSerializer.save(ds, str(outfile))

    LOG.warning('%d erroneous molecules', validator.n_errors)


if __name__ == '__main__':
    main()
