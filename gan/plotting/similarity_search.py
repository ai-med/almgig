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
from typing import Dict
import numpy as np
from rdkit import Chem, DataStructs
from guacamol.utils.chemistry import get_fingerprints
from sklearn.neighbors import NearestNeighbors

from ..hooks import CollectTensorHook
from ..mol.alice.data import PredictionInputFunction
from ..mol.alice.predict import collect_predictions
from ..mol.data.graph2mol import get_dataset, get_decoder
from ..mol.predict import load_estimator_from_dir
from .interpolate_embedding_grid import draw_svg_image

LOG = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Path to directory with checkpoints.')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Path to directory with pickled graphs.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Path to write SVG to.')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples from test data.')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='Number of nearest neighbors.')
    return parser


def nearest_neighbor_search(data: Dict[str, Dict[str, np.ndarray]],
                            n_neighbors: int,
                            n_samples: int,
                            outdir: Path):
    data_train = data['train']
    data_test = data['test']
    data_test['embedding'] = data_test['embedding'][:n_samples]
    data_test['nodes'] = data_test['nodes'][:n_samples]
    data_test['edges'] = data_test['edges'][:n_samples]

    knn = NearestNeighbors(n_neighbors=n_neighbors,
                           metric='cosine',
                           n_jobs=4)
    knn.fit(data_train['embedding'])
    dist, ind = knn.kneighbors(data_test['embedding'], return_distance=True)

    conv = get_decoder('gdb9', strict=True)
    mols_train = [conv.to_mol(n, e)
                  for n, e in zip(data_train['nodes'], data_train['edges'])]
    fps_train = get_fingerprints(mols_train)

    mols_test = [conv.to_mol(n, e)
                 for n, e in zip(data_test['nodes'], data_test['edges'])]
    fps_test = get_fingerprints(mols_test)

    for i, neighbors in enumerate(zip(dist, ind)):
        mol_test = mols_test[i]
        mols_emb = [mol_test]
        values_emb = []
        fps_emb = []

        for d, train_idx in zip(*neighbors):
            values_emb.append(d)
            mols_emb.append(mols_train[train_idx])
            fps_emb.append(fps_train[train_idx])

        sim_emb = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_test[i], fps_emb))

        mols_fp = [mol_test]
        values_fp = []
        sim_fp = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_test[i], fps_train))
        order = np.argsort(-sim_fp, kind='mergesort')  # sort in descending order
        for idx in order[:n_neighbors]:
            mols_fp.append(mols_train[idx - 1])
            values_fp.append([sim_fp[idx]])

        img = draw_svg_image(mols_emb, zip(values_emb, sim_emb), size=100)
        with (outdir / '{:04d}-emb.svg'.format(i)).open('w') as fout:
            fout.write(img)
        img = draw_svg_image(mols_fp, values_fp, size=100)
        with (outdir / '{:04d}-fp.svg'.format(i)).open('w') as fout:
            fout.write(img)

        with (outdir / '{:04d}-smiles.csv'.format(i)).open('w') as fout:
            fout.write("NeighborsByEmbeddingDistance,")
            fout.writelines(",".join([Chem.MolToSmiles(m) for m in mols_emb]))
            fout.write("\nCosineDistance,,")
            fout.writelines(",".join(map(str, values_emb)))
            fout.write("\nTanimotoSimilarity,,")
            fout.writelines(",".join(map(str, sim_emb)))
            fout.write("\nNeighborsByTanimotoSimilarity,")
            fout.writelines(",".join([Chem.MolToSmiles(m) for m in mols_fp]))
            fout.write("\nTanimotoSimilarity,,")
            fout.writelines(",".join(map(lambda x: str(x[0]), values_fp)))


def compute_embedding(model_dir: Path,
                      data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with (model_dir / 'args.pkl').open('rb') as fp:
        margs = pickle.load(fp)

    data_params = get_dataset(margs.dataset)

    estimator = load_estimator_from_dir(model_dir, batch_size=128)

    data = {}
    for kind in ('train', 'test'):
        input_fact = PredictionInputFunction(data_dir,
                                             num_latent=margs.num_latent,
                                             data_params=data_params,
                                             kind=kind)
        collect_edges = CollectTensorHook('adjacency_in:0')
        collect_nodes = CollectTensorHook('features:0')

        input_fn = input_fact.create(n_samples=None, batch_size=128)
        predictions = estimator.predict(input_fn,
                                        predict_keys='embedding',
                                        hooks=[collect_nodes, collect_edges])
        pred = collect_predictions(predictions)
        data[kind] = {
            'embedding': np.row_stack(pred['embedding']),
            'nodes': np.row_stack(collect_nodes.data),
            'edges': np.row_stack(collect_edges.data)
        }
        LOG.info('%s: retrieved embeddings %s', kind, data[kind]['embedding'].shape)

    return data


def main():
    parser = _create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = compute_embedding(args.model_dir, args.data_dir)
    if not args.output.exists():
        args.output.mkdir(parents=True)
    nearest_neighbor_search(data,
                            args.n_neighbors,
                            args.n_samples,
                            args.output)


if __name__ == '__main__':
    main()
