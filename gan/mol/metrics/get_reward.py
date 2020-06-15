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
from rdkit import Chem
from tqdm import tqdm

from gan.mol.metrics.base import GraphMolecularMetrics, MolecularMetrics, RewardsFactory, RewardType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles_file',
                        help='Path to file with molecules in SMILES format.')
    parser.add_argument('--reward_type', type=RewardType.from_string,
                        metavar=RewardType.metavar(),
                        required=True)
    parser.add_argument('--norm_file',
                        help='Path to file to standardize penalized logP score.')
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    factory = RewardsFactory(args.reward_type,
                             GraphMolecularMetrics._SA_MODEL,
                             GraphMolecularMetrics._NP_MODEL,
                             args.norm_file)
    metrics = MolecularMetrics(factory)

    inputs = []

    def _load_data():
        with open(args.smiles_file) as fin:
            for line in fin:
                smi = line.strip()
                mol = Chem.MolFromSmiles(smi)
                yield mol
                inputs.append(smi)

    rewards = metrics.get_reward_metrics(tqdm(_load_data())).squeeze()

    with open(args.output, 'w') as fout:
        header = ',SMILES,{}\n'.format(args.reward_type.name)
        fout.write(header)
        for i, (s, r) in enumerate(zip(inputs, rewards)):
            fout.write('{},{},{:f}\n'.format(i, s, r))


if __name__ == '__main__':
    main()
