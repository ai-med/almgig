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
import numpy as np
import pandas as pd
from guacamol.utils.chemistry import get_fingerprints
from rdkit import Chem, DataStructs
from tqdm import tqdm

from gan.plotting.interpolate_embedding_grid import draw_svg_image

LOG = logging.getLogger(__name__)


def read_smiles(filename):
    with open(filename) as fin:
        for line in fin:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            yield mol


def load_predict_file(predict_file):
    pred_data = pd.read_csv(predict_file, usecols=['SMILES', 'is_novel'], squeeze=True)
    LOG.info('Read %d molecules from %s', pred_data.shape[0], predict_file)
    pred_data = pred_data.query('is_novel == 1')
    LOG.info('%d molecules are novel', pred_data.shape[0])

    mols_pred = [Chem.MolFromSmiles(smi)
                 for _, smi in pred_data['SMILES'].iteritems()]
    fp_pred = get_fingerprints(mols_pred)

    return fp_pred, mols_pred


def compute_similarity(fp_pred, fp_train, mols_pred, mols_train):
    results = []
    for i, fp in enumerate(tqdm(fp_pred)):
        dist = DataStructs.BulkTanimotoSimilarity(fp, fp_train)
        idx = int(np.argmax(dist))
        results.append((mols_pred[i], mols_train[idx], dist[idx]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--train_file', type=Path, required=True)
    parser.add_argument('--predict_file', type=Path, required=True)
    parser.add_argument('-n', '--n_samples', type=int, default=10)
    parser.add_argument('--sort', choices=['ascending', 'descending'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    mols_train = list(read_smiles(args.train_file))
    LOG.info('Read %d molecules from %s', len(mols_train), args.train_file)
    fp_train = get_fingerprints(mols_train)

    fp_pred, mols_pred = load_predict_file(args.predict_file)
    if not args.sort:
        fp_pred = fp_pred[:args.n_samples]

    pkl = Path(args.predict_file.parent / (args.predict_file.stem + '_similarity.pkl'))
    if pkl.exists():
        LOG.info('Loading similarity from %s', pkl)
        with open(pkl, 'rb') as fin:
            results = pickle.load(fin)
    else:
        results = compute_similarity(fp_pred, fp_train, mols_pred, mols_train)
        LOG.info('Writing similarity to %s', pkl)
        with open(pkl, 'wb') as fin:
            pickle.dump(results, fin)

    if args.sort:
        results = sorted(results, key=lambda x: x[-1], reverse=args.sort == 'descending')

    LOG.info('Average similarity is %.4f', np.mean([v[2] for v in results]))

    for i, (mol_ref, mol_train, sim) in enumerate(results[:args.n_samples]):
        img = draw_svg_image([mol_ref, mol_train], [[sim]], 100, x_start=1)
        outpath = args.output / 'sim-{:03d}.svg'.format(i)
        with outpath.open('w') as fout:
            fout.write(img)


if __name__ == '__main__':
    main()
