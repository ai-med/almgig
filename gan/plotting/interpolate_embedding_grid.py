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
from guacamol.utils.chemistry import get_fingerprints
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
import scipy.sparse as sp

from ..mol.data.graph2mol import GDB9, get_decoder
from ..mol.data.serialize import graph_to_adj
from ..mol.predict import load_estimator_from_dir
from ..mol.alice.data import EmbedAndReconstructInputFunction, GenerateInputFunction


LOG = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Path to directory with checkpoints.')
    parser.add_argument('--graph_file', type=Path, required=True,
                        help='Path to file with pickled graphs.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Path to write SVG to.')
    parser.add_argument('--seed', type=int, default=5715,
                        help='Random number seed.')
    parser.add_argument('--sample_a_smiles',
                        help='SMILES of first sample.')
    parser.add_argument('--sample_b_smiles',
                        help='SMILES of second sample.')
    parser.add_argument('--n_interpolate', type=int, default=8,
                        help='Number of interpolation points.')
    parser.add_argument('--with_distance', action='store_true', default=False,
                        help='Whether to add Tanimoto distance to molecule A.')
    parser.add_argument('--image_size', type=int, default=100,
                        help='Height and weight of each molecule image.')
    return parser


def graph_to_padded_adj(graph, max_length):
    _, adj, _, _ = graph_to_adj(graph)

    shape = (max_length, max_length)
    if not isinstance(adj, sp.coo_matrix):
        adj = adj.tocoo()
    nmat = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                         dtype=adj.dtype,
                         shape=shape)
    return nmat.toarray()


def get_data(filename, smiles, rnd):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)

    if smiles is None:
        idx = rnd.randint(len(data))
        graph = data[idx]
        LOG.info('Retrieved graph at index %d: %s', idx, graph.graph)
    else:
        data_s = {g.graph['smiles']: g for g in data}
        graph = data_s[smiles]
        LOG.info('Retrieved graph: %s', graph.graph)

    edges = graph_to_padded_adj(graph, GDB9.MAX_NODES)

    nodes = []
    for n, v in graph.nodes(data='AtomCode'):
        assert v is not None
        nodes.append(v)
    for i in range(len(nodes), GDB9.MAX_NODES):
        nodes.append(0)

    nodes = np.array(nodes)

    # add batch dimension
    return nodes[np.newaxis], edges[np.newaxis]


def interpolate_in_latent_space(estimator, graph_1, graph_2, n_generate, num_latent, seed):
    embed_factory = EmbedAndReconstructInputFunction(num_latent=num_latent)

    nodes_1, edges_1 = graph_1
    predict_fn = embed_factory.create(nodes_1, edges_1, seed)
    for pred in estimator.predict(predict_fn, predict_keys='embedding'):
        z1 = pred['embedding']

    nodes_2, edges_2 = graph_2
    predict_fn = embed_factory.create(nodes_2, edges_2, seed)
    for pred in estimator.predict(predict_fn, predict_keys='embedding'):
        z2 = pred['embedding']

    z_list = [z1]
    d = z2 - z1
    increment = 1.0 / (n_generate + 1)
    multiplier = increment
    while multiplier < 1.0:
        z_list.append(z1 + multiplier * d)
        multiplier += increment
    z_list.append(z2)

    z_mat = np.row_stack(z_list).astype(np.float32)

    gen_factory = GenerateInputFunction(num_latent=num_latent)
    gen_fn = gen_factory.create(z_mat, seed)

    yield nodes_1[0], edges_1[0]
    for pred in estimator.predict(gen_fn):
        yield pred['features'], pred['adjacency']
    yield nodes_2[0], edges_2[0]


def compute_similarity(mols):
    fps = get_fingerprints(mols)
    arr = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
    return np.array(arr)


TEXT_TEMPL = """<text
   x="{x_coord:.2f}"
   y="{y_coord:.2f}"
   style="font-style:normal;font-weight:normal;font-size:12px;font-family:sans-serif;text-anchor:middle;fill:#000000;fill-opacity:1;stroke:none;text-align:center;">
   {lines}</text>"""

LINE_START = '<tspan '
LINE_BODY = 'x="{x_coord:.2f}" dy="{dy:.2f}em" style="text-anchor:middle;text-align:center;">{value:.3f}'
LINE_END = '</tspan>'
LINE_DELIM = LINE_END + LINE_START


class DropBackgroundAndAddHeightMapFn:

    def __init__(self, height, extra_height):
        self._height_set = False
        self._pattern = "height='{:d}px'".format(height)
        self._repl = "height='{:d}px'".format(height + extra_height)

    def __call__(self, line):
        if line.startswith("<rect ") and line.endswith("</rect>"):
            return ""
        if not self._height_set and self._pattern in line:
            self._height_set = True
            return line.replace(self._pattern, self._repl)

        return line


def draw_svg_image(mol_list, similarity, size, x_start=1.5):
    img = Draw.MolsToGridImage(mol_list,
                               molsPerRow=len(mol_list),
                               subImgSize=(size, size),
                               useSVG=True)
    lines = img.split('\n')
    x = x_start * size
    y = size - 0.1 * size
    text = []

    for item in similarity:
        dy = 0.8
        texts = [LINE_START]
        for i, value in enumerate(item):
            l = LINE_BODY.format(x_coord=x, dy=dy, value=value)
            texts.append(l)
            texts.append(LINE_DELIM)
            dy += 0.2
        texts[-1] = LINE_END
        t = TEXT_TEMPL.format(x_coord=x, y_coord=y, lines="\n".join(texts))
        text.append(t)
        x += size

    imgo = '\n'.join(map(DropBackgroundAndAddHeightMapFn(size, 15), lines[:-2] + text)) + '\n</svg>'
    return imgo


def main():
    logging.basicConfig(level=logging.WARNING)

    parser = _create_parser()
    args = parser.parse_args()

    rnd = np.random.RandomState(args.seed)
    filename = args.graph_file

    estimator, num_latent = load_estimator_from_dir(args.model_dir, return_num_latent=True, batch_size=1)

    graph_1 = get_data(filename, args.sample_a_smiles, rnd)
    graph_2 = get_data(filename, args.sample_b_smiles, rnd)
    n_generate = args.n_interpolate - 2

    conv = get_decoder('gdb9', strict=True)
    mol_list = []
    smiles = []
    for feat, adj in interpolate_in_latent_space(estimator, graph_1, graph_2, n_generate, num_latent, args.seed):
        try:
            mol = conv.to_mol(feat, adj)
            mol_list.append(mol)
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        except ValueError:
            LOG.warning('Converting graph to molecule failed')
            continue

    if args.with_distance:
        similarity = compute_similarity(mol_list)[:, np.newaxis]
    else:
        similarity = []
    img = draw_svg_image(mol_list, similarity, args.image_size)
    with args.output.open('w') as fout:
        fout.write(img)
    csv_file = args.output.parent / "{}.csv".format(args.output.stem)

    with csv_file.open('w') as fout:
        fout.write("SMILES,TanimotoSimilarity\n")
        for i, smi in enumerate(smiles):
            fout.write(smi)
            if len(similarity) > 0 and i > 0:
                fout.write(",{:.8f}".format(similarity[i - 1, 0]))
            fout.write("\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
