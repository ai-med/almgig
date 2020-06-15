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
import pandas as pd
from rdkit import Chem

from .interpolate_embedding_grid import draw_svg_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('-o', '--output')
    parser.add_argument('-c', '--column')
    parser.add_argument('-n', '--n_samples', type=int, default=5)
    parser.add_argument('--size', type=int, default=200)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--descending', dest='descending', action='store_true')
    g.add_argument('--ascending', dest='descending', action='store_false')

    args = parser.parse_args()

    data = pd.read_csv(args.csv_file)
    if args.column is not None:
        data.sort_values(by=args.column, ascending=not args.descending, inplace=True)
    data_sel = data.iloc[:args.n_samples]

    mols = [Chem.MolFromSmiles(smi)
            for _, smi in data_sel['SMILES'].iteritems()]

    if args.column is not None:
        labels = data_sel.loc[:, args.column]
    else:
        labels = []
    img = draw_svg_image(mols, labels, args.size,
                         x_start=0.5)
    with open(args.output, 'w') as fout:
        fout.write(img)


if __name__ == '__main__':
    main()
