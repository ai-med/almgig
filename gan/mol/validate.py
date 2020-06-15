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
from collections import defaultdict
from itertools import chain
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Union
from rdkit import Chem
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.chemistry import canonicalize_list
from guacamol.utils.sampling_helpers import sample_unique_molecules
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from .data import Graph2Mol
from .metrics import GraphMolecularMetrics
from .metrics.distribution_learning import assess_distribution_learning
from .metrics.scores import ValidityScore

LOG = logging.getLogger(__name__)

GenPredictFn = Callable[[int, int], Callable[[], tf.Tensor]]


class EstimatorGenerator(DistributionMatchingGenerator):

    def __init__(self,
                 estimator: tf.estimator.Estimator,
                 make_predict_fn: GenPredictFn,
                 graph_converter: Graph2Mol,
                 batch_size: int = 1000) -> None:
        self.estimator = estimator
        self.make_predict_fn = make_predict_fn
        self.graph_converter = graph_converter
        self.batch_size = batch_size

    def generate(self, number_samples: int) -> List[str]:
        samples = tqdm(self.iter_generate(number_samples),
                       desc='Sampling from estimator',
                       total=number_samples)
        return list(samples)

    def iter_generate(self, number_samples: int) -> Iterable[str]:
        LOG.info('Validating %s', self.estimator.latest_checkpoint())

        n = number_samples // self.batch_size
        iters = [self._predict_smiles(self.make_predict_fn(n * self.batch_size, self.batch_size))]
        if number_samples % self.batch_size != 0:
            size = number_samples - n * self.batch_size
            iters.append(self._predict_smiles(self.make_predict_fn(size, size)))
        return chain.from_iterable(iters)

    def _predict_smiles(self, predict_fn):
        conv = self.graph_converter
        for pred in self.estimator.predict(predict_fn):
            feat, adj = pred['features'], pred['adjacency']
            try:
                mol = conv.to_mol(feat, adj)
                smi = Chem.MolToSmiles(mol)
                if not ValidityScore.is_valid_smiles(smi):
                    smi = ''
            except ValueError:
                smi = ''
            yield smi


def save_metrics(model: EstimatorGenerator,
                 training_set_file: Union[str, Path],
                 output_file: Union[str, Path]) -> None:
    training_set = [s.strip() for s in open(training_set_file).readlines()]
    training_set_molecules = set(canonicalize_list(training_set, include_stereocenters=False))
    LOG.info('Loaded %d unique molecules from %s', len(training_set_molecules), training_set_file)

    metrics = GraphMolecularMetrics(None, None)
    gen_molecules = sample_unique_molecules(model, 10000)
    pbar = tqdm(gen_molecules,
                desc='Computing metrics',
                total=10000)

    indices = []
    samples = defaultdict(lambda: [])
    for i, smi in enumerate(pbar):
        if smi is None or not ValidityScore.is_valid_smiles(smi):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        values = metrics.get_validation_metrics([mol])
        values['SMILES'] = smi
        values['is_novel'] = 0 if smi in training_set_molecules else 1

        for key, val in values.items():
            if isinstance(val, list):
                assert len(val) == 1
                val = val[0]
            samples[key].append(val)
        indices.append(i)

    df = pd.DataFrame.from_dict(samples)
    df.index = indices
    LOG.info('Saving metrics to %s', output_file)
    df.to_csv(output_file)


def validate(estimator: tf.estimator.Estimator,
             predict_fn: GenPredictFn,
             graph_converter: Graph2Mol,
             train_data_file: str,
             output_file: str) -> None:
    model = EstimatorGenerator(estimator, predict_fn, graph_converter)

    out_path = Path(output_file)
    save_metrics(model, train_data_file, out_path.with_suffix('.csv'))

    assess_distribution_learning(model,
                                 training_file_path=train_data_file,
                                 json_output_file=output_file,
                                 number_samples=10000)
