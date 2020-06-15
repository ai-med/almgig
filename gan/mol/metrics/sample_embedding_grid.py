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
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from gan.mol.data.graph2mol import get_decoder
from gan.mol.alice.data import EmbedAndReconstructInputFunction, GenerateInputFunction
from gan.plotting.interpolate_embedding_grid import get_data
from gan.mol.predict import load_estimator_from_dir

LOG = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Path to directory with checkpoints.')
    parser.add_argument('--graph_file', type=Path, required=True,
                        help='Path to file with pickled graphs.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Path to write SVG to.')
    parser.add_argument('--output_smiles', type=Path,
                        help='Path to file to write SMILES to.')
    parser.add_argument('--seed', type=int, default=5715,
                        help='Random number seed.')
    parser.add_argument('--sample_smiles',
                        help='SMILES of first sample.')
    parser.add_argument('--axis_one', type=int, default=0,
                        help='First axis of embedding space to alter.')
    parser.add_argument('--axis_two', type=int, default=1,
                        help='Second axis of embedding space to alter.')
    parser.add_argument('--gap', type=float, default=2.,
                        help='Gap in embedding space between samples.')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples along each axis.')
    return parser


def retrieve_latent_code(estimator, graph, seed):
    nodes_1, edges_1 = graph
    fact = EmbedAndReconstructInputFunction(num_latent=64)
    predict_fn = fact.create(nodes_1, edges_1, seed)
    for pred in estimator.predict(predict_fn, predict_keys='embedding'):
        z1 = pred['embedding']
    return z1


def sample_in_latent_space(estimator, graph, axis, gap_size, n_generate, seed):
    z1 = retrieve_latent_code(estimator, graph, seed)

    offset = np.cumsum(np.ones(n_generate) * gap_size)
    offsets = np.concatenate((-offset[::-1], [0], offset))

    z_mat = []
    for x in offsets:
        for y in offsets:
            z = z1.copy()
            z[axis] += [x, y]
            z_mat.append(z)
    z_mat = np.stack(z_mat, axis=0)

    fact = GenerateInputFunction(num_latent=64)
    gen_fn = fact.create(z_mat, seed)
    for pred in estimator.predict(gen_fn):
        yield pred['features'], pred['adjacency']


def main():
    logging.basicConfig(level=logging.WARNING)

    parser = _create_parser()
    args = parser.parse_args()

    rnd = np.random.RandomState(args.seed)
    estimator, num_latent = load_estimator_from_dir(args.model_dir, return_num_latent=True, batch_size=1)
    graph = get_data(args.graph_file, args.sample_smiles, rnd)
    axis = [args.axis_one, args.axis_two]

    conv = get_decoder('gdb9', strict=True)
    mol_list = []
    for feat, adj in sample_in_latent_space(estimator, graph, axis, args.gap, args.n_samples, args.seed):
        try:
            mol = conv.to_mol(feat, adj)
            mol_list.append(mol)
        except ValueError:
            LOG.warning('Converting graph to molecule failed')
            mol_list.append(Chem.MolFromSmiles('C'))
            continue

    per_row = args.n_samples * 2 + 1
    img = Draw.MolsToGridImage(mol_list,
                               molsPerRow=per_row,
                               subImgSize=(100, 100),
                               useSVG=True)
    with args.output.open('w') as fout:
        fout.write(img)

    if args.output_smiles:
        with args.output_smiles.open('w') as fout:
            i = 1
            for m in mol_list:
                smi = Chem.MolToSmiles(m, isomericSmiles=True)
                fout.write(smi)
                if i == per_row:
                    fout.write("\n")
                    i = 1
                else:
                    fout.write("\t")
                    i += 1
            fout.write("\n")


if __name__ == '__main__':
    main()
