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
from collections import Counter
import logging
from pathlib import Path
import pickle
from typing import Optional, Iterable, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem

from gan.mol.data.graph2mol import get_decoder, get_dataset, Graph2Mol
from gan.mol.predict import load_estimator_from_dir
from gan.mol.alice.data import PredictionInputFunction

LOG = logging.getLogger(__name__)


def counter_to_data_frame(counter, n_samples):
    counter = pd.Series(counter)
    df = pd.DataFrame({'frequency': counter,
                       'percentage': counter * 100 / float(n_samples)})
    return df


def collect_errors(predict_iter: Iterable[Tuple[np.ndarray, np.ndarray]],
                   conv: Graph2Mol) -> pd.DataFrame:
    counter = {
        'non-atom-bond': 0,
        'valency': 0,
        'smiles-invalid': 0,
        'smiles-no-atom-symbol': 0,
        'smiles-dot': 0,
        'valid': 0,
    }
    components_counter = {1: 0}
    components_atoms_counter = {}

    n_samples = 0
    for feat, adj in predict_iter:
        n_samples += 1
        try:
            mol = conv.to_mol(feat, adj)
        except ValueError as e:
            counter['non-atom-bond'] += 1
            continue

        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
            counter['valency'] += 1
            continue

        smi = Chem.MolToSmiles(mol)
        if smi == '':
            counter['smiles-invalid'] += 1
            continue
        if '*' in smi:
            counter['smiles-no-atom-symbol'] += 1
            continue
        if '.' in smi:
            counter['smiles-dot'] += 1
            num_components = Counter(smi)['.'] + 1
            if num_components not in components_counter:
                components_counter[num_components] = 0
                components_atoms_counter[num_components] = Counter()
            components_counter[num_components] += 1
            for sub_smi in smi.split('.'):
                sub_mol = Chem.MolFromSmiles(sub_smi)
                components_atoms_counter[num_components].update([
                    sub_mol.GetNumAtoms()
                ])
            continue
        else:
            components_counter[1] += 1

        counter['valid'] += 1

    df_counter = counter_to_data_frame(counter, n_samples)
    df_components_counter = counter_to_data_frame(components_counter, n_samples)

    LOG.info("Num Atoms per number of components:\n%r\n", components_atoms_counter)

    return df_counter, df_components_counter


def get_error_statistics(model_dir: Path,
                         data_dir: Optional[Path] = None) -> pd.DataFrame:
    estimator = load_estimator_from_dir(model_dir, batch_size=100)
    with (model_dir / 'args.pkl').open('rb') as fp:
        args = pickle.load(fp)

    data_dir = data_dir or args.data_dir
    data_params = get_dataset(args.dataset)
    fact = PredictionInputFunction(data_dir=data_dir,
                                   num_latent=args.num_latent,
                                   data_params=data_params,
                                   kind='test')
    # choose a large limit so we iterate over the whole test data
    n_samples = args.batch_size * 1000
    predict_fn = fact.create(n_samples, args.batch_size)

    recon_data = []
    gen_data = []
    for pred in estimator.predict(predict_fn):
        recon = pred['reconstructed/features'], pred['reconstructed/adjacency']
        gen = pred['features'], pred['adjacency']
        recon_data.append(recon)
        gen_data.append(gen)

    conv = get_decoder(args.dataset, strict=False)
    dfs_recon = collect_errors(recon_data, conv)
    dfs_gen = collect_errors(gen_data, conv)

    dfs = []
    for df_recon, df_gen in zip(dfs_recon, dfs_gen):
        df_recon.columns = pd.MultiIndex.from_arrays(
            [['reconstructed'] * df_recon.shape[1],
            df_recon.columns])
        df_gen.columns = pd.MultiIndex.from_arrays(
            [['generated'] * df_gen.shape[1],
            df_gen.columns])

        df = pd.concat((df_recon, df_gen), axis=1)
        dfs.append(df)
    return dfs


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--latex', action='store_true', default=True)

    args = parser.parse_args()

    dfs = get_error_statistics(args.model_dir, args.data_dir)
    if args.output:
        with args.output.open('w') as fout:
            for df in dfs:
                if args.latex:
                    txt = df.round(2).to_latex(multicolumn=True)
                    fout.write(txt)
                else:
                    df.to_csv(fout)
                fout.write("\n\n")
    else:
        print(dfs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
