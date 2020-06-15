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
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw

LOG = logging.getLogger(__name__)

ATOM_MAP = np.array(['C', 'F', 'H', 'N', 'O'])
VALENCE_MAP = np.array([4, 1, 1, 3, 2], dtype=np.intp)


def to_mol(nodes: np.ndarray,
           edges: np.ndarray) -> Chem.rdchem.Mol:
    mol = Chem.RWMol()
    for nt in ATOM_MAP[nodes]:
        mol.AddAtom(Chem.Atom(nt))

    for i, j, k in edges:
        if i == j:
            continue
        mol.AddBond(int(i), int(j), Chem.rdchem.BondType.values[k])

    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def get_edge_candidates(num_nodes: int,
                        num_edges_types: int) -> List[Tuple[int, int, int, int]]:
    indices = np.tril_indices(num_nodes, k=0)
    list_edges = np.empty(indices[0].shape[0] * num_edges_types, dtype=object)
    ii = 0
    for flat_idx, (i, j) in enumerate(zip(*indices)):
        for k in range(num_edges_types):
            list_edges[ii] = (flat_idx, int(i), int(j), k + 1)
            ii += 1
    return list_edges


def sample_molecule_graph(num_edges: int,
                          feature_probs: np.ndarray,
                          edge_logits: np.ndarray,
                          weight_logits: np.ndarray,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    assert feature_probs.ndim == 2
    assert edge_logits.ndim == 1
    assert weight_logits.ndim == 2
    assert edge_logits.shape[0] == weight_logits.shape[0]

    num_nodes, num_node_types = feature_probs.shape
    n_edges, n_edge_types = weight_logits.shape
    rnd = np.random.RandomState(seed)

    LOG.debug("Sampling molecule with %d nodes and %d edges of %d types",
              num_nodes, num_edges, n_edge_types)

    # sample nodes
    node_ids = np.arange(num_node_types, dtype=np.intp)
    sampled_nodes = np.asarray([rnd.choice(node_ids, p=f) for f in feature_probs])

    # initialize based on valence of sampled node features
    assert n_edge_types < VALENCE_MAP.max()
    valence = VALENCE_MAP[sampled_nodes]
    degree = np.zeros(num_nodes)
    allowed_valence = np.ones((num_nodes, n_edge_types))
    for i, left in enumerate(valence):
        allowed_valence[i, left:] = 0

    allowed_weights = np.ones_like(weight_logits)
    indices = np.tril_indices(num_nodes)
    for k, (row, col) in enumerate(zip(*indices)):
        if row == col:
            allowed_weights[k, :] = 0
        else:
            allowed_weights[k, :] = allowed_valence[row] * allowed_valence[col]

#     num_nodes = int(-0.5 + np.sqrt(0.25 + 2 * edge_logits.shape[0]))
#     assert num_nodes * (num_nodes + 1) // 2 == edge_logits.shape[0]

    # sample edges
    list_edges = get_edge_candidates(num_nodes, n_edge_types)
    assert len(list_edges) == n_edges * n_edge_types
    edge_ids = np.arange(len(list_edges), dtype=np.intp)
    eps = np.finfo(edge_logits.dtype).eps

    edge_logits_s = edge_logits.copy()
    edge_logits_s -= np.max(edge_logits_s)
    prob_edge = np.exp(edge_logits_s)

    weight_logits_s = weight_logits.copy()
    weight_logits_s -= np.max(weight_logits_s, axis=1, keepdims=True)
    prob_weight = np.exp(weight_logits_s)

    sampled_edges = []
    unselected_edges = np.ones(n_edges, dtype=np.bool_)
    for _ in range(num_edges):
        prob_edge[~unselected_edges] = eps
        prob_edge /= np.sum(prob_edge)
        assert np.allclose(prob_edge.sum(), 1.0)

        selected_weights = ~(
                allowed_weights * unselected_edges[:, np.newaxis].astype(allowed_weights.dtype)
            ).astype(np.bool_)
        prob_weight[selected_weights] = eps
        prob_weight /= np.sum(prob_weight, axis=1, keepdims=True)
        assert np.allclose(prob_weight.sum(1), 1.0)

        prob_comb = prob_weight * prob_edge[:, np.newaxis]
        # account for cases where any edge (no matter its weight) isn't allowed
        prob_comb[selected_weights] = 0.0
        # order of elements needs to match list_edges
        prob_comb = prob_comb.flatten()
        s = prob_comb.sum()
        if np.allclose(s, 0.0):
            LOG.debug("No edges left to sample from. Aborting.")
            break
        prob_comb /= s

        sampled_id = rnd.choice(edge_ids, p=prob_comb, replace=False)
        sel_edge = list_edges[sampled_id]
        sampled_edges.append(sel_edge[1:])

        # idx_edge is the flat index
        idx_edge, idx_u, idx_v, weight = sel_edge
        unselected_edges[idx_edge] = False

        degree[idx_u] += weight
        degree[idx_v] += weight

        d = (valence - degree).astype(int)
        for i, left in enumerate(d):
            allowed_valence[i, left:] = 0

        for k, (row, col) in enumerate(zip(*indices)):
            if row != col:
                allowed_weights[k, :] = allowed_valence[row] * allowed_valence[col]

    return sampled_nodes, sampled_edges


def sample_molecule(pred, num_edges, num_trials):
    for _ in range(num_trials):
        nodes, edges = sample_molecule_graph(num_edges,
                                             pred['prob_features'],
                                             pred['logits_edge'].squeeze(),
                                             pred['logits_edge_weight'])
        # print(ATOM_MAP[nodes])
        # for iu, iv, m in edges:
        #     u = ATOM_MAP[nodes[iu]]
        #     v = ATOM_MAP[nodes[iv]]
        #     print(f"{u}({iu})-{m}-{v}({iv})")
        mol = to_mol(nodes, edges)
        yield mol


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path,
                        default=Path('./predictions.pkl'),
                        help='Path to pickle file with predictions.')
    parser.add_argument('-o', '--output', type=Path,
                        default='samples.csv',
                        help='Path to CSV file to write output to.')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of trials to sample molecule for each prediction.')
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args=args)

    with args.input.open('rb') as fin:
        predictions = pickle.load(fin)

    pbar = tqdm(total=len(predictions) * args.num_trials,
                desc='Sampling molecules', unit=' molecules')
    with args.output.open('w') as fout:
        for i, pred in enumerate(predictions):
            num_edges = int(np.round(pred['num_edges'], 0))
            mol_iter = sample_molecule(pred, num_edges, args.num_trials)
            for j, mol in enumerate(mol_iter):
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                txt = f'{i},{j},{smi}\n'
                fout.write(txt)
                pbar.update()


if __name__ == '__main__':
    main()
