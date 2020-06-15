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
import pickle
import numpy as np
from tensorpack.dataflow import LMDBData, MapData
from tensorpack.utils.serialize import loads_msgpack
from gan.mol.data.graph2mol import get_decoder
from gan.mol.metrics import GraphMolecularMetrics
from gan.mol.metrics.scores import CycleLengthScore, LogPScore, SAScore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdb_file', required=True)
    parser.add_argument('--data', choices=['gdb9', 'zinc'], required=True,
                        help='Dataset to use.')
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    ds = LMDBData(args.mdb_file, shuffle=False)
    ds = MapData(ds, lambda dp: loads_msgpack(dp[1]))

    # used by PenalizedLogPScore
    ss = (CycleLengthScore(), LogPScore(), SAScore(GraphMolecularMetrics._SA_MODEL))
    conv = get_decoder(args.data, True)
    values = []
    for dp in ds.get_data():
        m = conv.to_mol(dp[1].squeeze(), dp[0])
        row = np.empty(len(ss), dtype=float)
        for i, s in enumerate(ss):
            row[i] = s.compute(m)
        values.append(row)

    values = np.row_stack(values)

    m = np.mean(values, axis=0)
    amin = np.min(values, axis=0)
    amax = np.max(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)

    out = {}
    for s, mv, sdv, mi, mx in zip(ss, m, sd, amin, amax):
        out[s.name] = {'mean': mv, 'std': sdv, 'min': mi, 'max': mx}

    with open(args.output, 'wb') as fout:
        pickle.dump(out, fout)
