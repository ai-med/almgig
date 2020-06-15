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
from pathlib import Path
from typing import Iterable, Union
import numpy as np
from rdkit import Chem
from sklearn.utils.validation import check_random_state
from tqdm import tqdm

from nevae.sample import sample_molecule_graph, to_mol


def softmax(x, axis=None):
    amax = np.max(x, axis=axis, keepdims=True)
    xx = np.exp(x - amax)
    return xx / np.sum(xx, axis=axis, keepdims=True)


class RandomGenerator:

    def __init__(self,
                 n_nodes: int,
                 n_node_types: int,
                 n_edge_types: int,
                 random_state: Union[np.random.RandomState, int, None] = None) -> None:
        self.n_nodes = n_nodes
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.random_state = random_state
        # C, F, H, N, O
        self._atoms = np.arange(self.n_node_types, dtype=int)
        # number of elements in the lower triangle, including main diagonal
        self._n_adj_flat = n_nodes * (n_nodes + 1) // 2

    def generate(self, n_samples: int) -> Iterable[Chem.rdchem.Mol]:
        rnd = check_random_state(self.random_state)
        for _ in range(n_samples):
            smi = self.generate_one(rnd)
            yield smi

    def generate_one(self, rnd: np.random.RandomState) -> Chem.rdchem.Mol:
        node_logits = rnd.randn(self.n_nodes, self.n_node_types)
        node_probs = softmax(node_logits, axis=1)

        edge_logits = rnd.randn(self._n_adj_flat)
        weight_logits = rnd.randn(self._n_adj_flat, self.n_edge_types)
        num_edges = self._n_adj_flat  # set to maximum

        sampled_nodes, sampled_edges = sample_molecule_graph(
            num_edges=num_edges,
            feature_probs=node_probs,
            weight_logits=weight_logits,
            edge_logits=edge_logits
        )
        mol = to_mol(sampled_nodes, sampled_edges)
        return mol


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path,
                        default='samples.csv',
                        help='Path to CSV file to write output to.')
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args=args)

    nodes = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
    n_samples = 5000

    pbar = tqdm(total=len(nodes) * n_samples,
                desc='Sampling molecules', unit=' molecules')
    with args.output.open('w') as fout:
        for i, n in enumerate(nodes):
            g = RandomGenerator(n_nodes=n,
                                n_node_types=5,
                                n_edge_types=3,
                                random_state=123)
            mol_iter = g.generate(n_samples)
            for j, mol in enumerate(mol_iter):
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                txt = f'{i},{j},{smi}\n'
                fout.write(txt)
                pbar.update()


if __name__ == '__main__':
    main()
