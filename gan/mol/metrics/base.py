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
import enum
from os.path import join, dirname
import numpy as np
from rdkit import Chem

from .scores import CombinedScore, QEDScore, LogPScore, MolecularWeightScore, \
    NormalizedScore, NPScore, PenalizedLogPScore, SAScore, ValidityScore


@enum.unique
class RewardType(enum.Enum):
    DRUGLIKELINESS = 1
    SYNTHESIZABILITY = 2
    PENALIZEDLOGP = 3
    ALL = 100

    @staticmethod
    def choices():
        return sorted([name.lower() for name in RewardType.__members__.keys()])

    @staticmethod
    def metavar():
        msg = ','.join(RewardType.choices())
        return '{%s}' % msg

    @staticmethod
    def from_string(str_value):
        from argparse import ArgumentTypeError

        try:
            return RewardType[str_value.upper()]
        except KeyError as e:
            msg = 'use one of ' + RewardType.metavar()
            raise ArgumentTypeError(msg)


class RewardsFactory:

    def __init__(self, reward_type, sa_model_file, np_model_file, standardization_file):
        self._reward_type = reward_type
        # cache classes, because they need to read a large file
        self._sa_score = SAScore(sa_model_file)
        self._np_score = NPScore(np_model_file)
        self._std_file = standardization_file

        self._reward_scores = None
        self._val_scores = None

    def _create_reward_scores(self):
        if self._reward_type == RewardType.DRUGLIKELINESS:
            reward_score = QEDScore()
        elif self._reward_type == RewardType.SYNTHESIZABILITY:
            reward_score = NormalizedScore(self._sa_score, 1.0)
        elif self._reward_type == RewardType.PENALIZEDLOGP:
            reward_score = PenalizedLogPScore(self._sa_score._sa_model_file, self._std_file)
        elif self._reward_type == RewardType.ALL:
            rewards = [NormalizedScore(self._sa_score, 1.0),
                       QEDScore()]
            reward_score = CombinedScore(rewards)
        else:
            raise ValueError('invalid reward type: {!r}'.format(self._reward_type))
        self._reward_scores = (ValidityScore(), reward_score)

    def get_reward_scores(self):
        if self._reward_scores is None:
            self._create_reward_scores()
        return self._reward_scores

    def _create_validation_scores(self):
        self._val_scores = (
            self._sa_score,
            self._np_score,
            LogPScore(),
            QEDScore(),
            MolecularWeightScore(),
        )

    def get_validation_scores(self):
        if self._val_scores is None:
            self._create_validation_scores()
        return self._val_scores


class MolecularMetrics:
    def __init__(self, rewards_factory, input_converter=None):
        self._rewards_factory = rewards_factory
        self._input_convert = input_converter

    def _as_molecule(self, inputs):
        if self._input_convert is None or inputs is None or isinstance(inputs, Chem.rdchem.Mol):
            return inputs
        return self._input_convert(inputs)

    def _safe_apply(self, func, inputs, error_score, **kwargs):
        mol = self._as_molecule(inputs)
        if mol is None:
            ret = error_score
        else:
            ret = func(mol, **kwargs)
        return ret

    def is_valid(self, mol):
        return self._safe_apply(ValidityScore().compute, mol, False)

    def _safe_apply_all(self, mols, scorers, raise_errors):
        values = defaultdict(lambda: [])
        for mol in map(self._as_molecule, mols):
            if mol is None and raise_errors:
                raise ValueError('invalid molecule')

            for scorer in scorers:
                if mol is None:
                    score = scorer.min_score
                else:
                    score = scorer.compute(mol)
                values[scorer.name].append(score)
        return values

    def get_reward_metrics(self, mols):
        scorers = self._rewards_factory.get_reward_scores()
        scores = self._safe_apply_all(mols, scorers, raise_errors=False)
        val = np.array(scores[scorers[0].name], dtype=np.int32)
        sa = np.array(scores[scorers[1].name], dtype=np.float32)
        rewards = val * sa
        return rewards[:, np.newaxis]

    def _get_validation_metrics(self, mols):
        validator = ValidityScore()
        valid_smiles = []
        valid_mols = []
        mol_count = 0
        for mol in mols:
            mol_count += 1
            mol = self._as_molecule(mol)
            if mol is None:
                continue
            s = validator.to_valid_smiles(mol)
            if s is not None:
                valid_smiles.append(s)
                valid_mols.append(mol)
        assert mol_count > 0

        values = self._safe_apply_all(valid_mols, self._rewards_factory.get_validation_scores(), raise_errors=True)
        return values, mol_count, valid_smiles

    def get_validation_metrics(self, mols):
        values, _, _ = self._get_validation_metrics(mols)
        return values

    def get_validation_metrics_summary(self, mols):
        values, mol_count, valid_smiles = self._get_validation_metrics(mols)

        n_valid = len(valid_smiles)
        if n_valid == 0:
            prop_unique = 0.0
        else:
            n_unique = len(set(valid_smiles))
            prop_unique = n_unique / n_valid
        results = {
            'validity': len(valid_smiles) / mol_count,
            'proportion_unique': prop_unique
        }
        for nam, scores in values.items():
            results[nam + '/mean'] = np.mean(scores)

        return results


class GraphMolecularMetrics(MolecularMetrics):
    # https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/fpscores.pkl.gz
    _SA_MODEL = join(dirname(__file__), 'data', 'SA_score.pkl.gz')
    # https://github.com/rdkit/rdkit/blob/master/Contrib/NP_Score/publicnp.model.gz
    _NP_MODEL = join(dirname(__file__), 'data', 'NP_score.pkl.gz')

    def __init__(self, graph_conv, reward_type, standardization_file=None):
        super(GraphMolecularMetrics, self).__init__(
            RewardsFactory(
                reward_type=reward_type,
                sa_model_file=self._SA_MODEL,
                np_model_file=self._NP_MODEL,
                standardization_file=standardization_file),
            input_converter=self._arrays_to_mol)

        self.conv = graph_conv

    def _arrays_to_mol(self, inputs):
        node_labels, edge_labels = inputs
        try:
            return self.conv.to_mol(node_labels, edge_labels)
        except ValueError:
            return None

    def metrics_from_arrays(self, node_labels, edge_labels):
        return self.get_validation_metrics_summary(zip(node_labels, edge_labels))
