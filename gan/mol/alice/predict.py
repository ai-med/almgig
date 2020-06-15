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
from collections import defaultdict, namedtuple
from pathlib import Path
import pickle
from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import tensorflow as tf

from gan.mol.data import Graph2Mol
from gan.mol.metrics import GraphMolecularMetrics

from ..data.graph2mol import onehot_to_dense
from ...hooks import CollectTensorHook
from ...utils import save_as_embedding

MoleculeGraph = namedtuple('MoleculeGraph',
                           ('nodes', 'edges'))

MoleculePredictions = namedtuple('MoleculePredictions',
                                 ('inputs', 'embeddings', 'reconstructions'))


def collect_predictions(predictions):
    data = defaultdict(lambda: [])
    for pred in predictions:
        for k, v in pred.items():
            data[k].append(v)

    return data


class EmbeddingSaver:

    def __init__(self,
                 embedding_dir: str,
                 reward_type: str,
                 graph_converter: Graph2Mol) -> None:
        self._embedding_dir = Path(embedding_dir)

        self._conv = graph_converter
        self._reward_type = reward_type

    def _get_predictions(self,
                         estimator: tf.estimator.Estimator,
                         eval_fn: Callable[[], Dict[str, tf.Tensor]]) -> MoleculePredictions:
        collect_edges = CollectTensorHook('adjacency_in:0')
        collect_nodes = CollectTensorHook('features:0')

        predictions = estimator.predict(eval_fn, hooks=[collect_edges, collect_nodes])
        pred = collect_predictions(predictions)

        feat = np.stack(pred['reconstructed/features'], axis=0)
        adj = np.stack(pred['reconstructed/adjacency'], axis=0)

        feat, adj = onehot_to_dense(feat, adj)

        mols_recon = MoleculeGraph(nodes=feat, edges=adj)

        mols_real = MoleculeGraph(
            nodes=np.row_stack(collect_nodes.data),
            edges=np.row_stack(collect_edges.data))

        return MoleculePredictions(inputs=mols_real,
                                   embeddings=np.row_stack(pred['embedding']),
                                   reconstructions=mols_recon)

    def _make_metrics_metadata(self, mols: List[Chem.rdchem.Mol]) -> pd.DataFrame:
        metrics = GraphMolecularMetrics(self._conv, self._reward_type)

        metrics_real = metrics.get_validation_metrics(mols)
        metrics_real['SMILES'] = [Chem.MolToSmiles(m) for m in mols]
        metrics_data = pd.DataFrame(metrics_real)

        out_file = self._embedding_dir / 'metrics.tsv'
        metrics_data.to_csv(out_file, index=False, sep="\t")
        return metrics_data

    def _make_inputs_sprite_image(self, mols_real: List[Chem.rdchem.Mol]) -> None:
        n_sprites = int(np.ceil(np.sqrt(len(mols_real))))
        image = Draw.MolsToGridImage(mols_real, n_sprites, subImgSize=(125, 125))
        out_file = self._embedding_dir / 'inputs.png'
        image.save(str(out_file), format='PNG')

    def save_as_checkpoint(self,
                           estimator: tf.estimator.Estimator,
                           eval_fn: Callable[[], Dict[str, tf.Tensor]]):
        pred = self._get_predictions(estimator, eval_fn)

        if not self._embedding_dir.exists():
            self._embedding_dir.mkdir(parents=True)

        mols_real = [self._conv.to_mol(n, e)
                     for n, e in zip(pred.inputs.nodes, pred.inputs.edges)]
        self._make_inputs_sprite_image(mols_real)
        metrics_data = self._make_metrics_metadata(mols_real)

        assert metrics_data.shape[0] == len(mols_real), '{} != {}'.format(
            metrics_data.shape[0], len(mols_real))

        cpkt_file = self._embedding_dir / 'data'
        save_as_embedding(pred.embeddings,
                          str(cpkt_file),
                          metadata_path='metrics.tsv',
                          sprite_image_path='inputs.png',
                          sprite_image_size=(125, 125))

        with (self._embedding_dir / 'predictions.pkl').open('wb') as fout:
            pickle.dump(pred, fout)
