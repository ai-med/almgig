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
from abc import ABCMeta, abstractmethod
import gzip
import pickle
import math
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors

from ..data.graph2mol import disable_rdkit_log

__all__ = [
    'CycleLengthScore',
    'QEDScore',
    'LogPScore',
    'MolecularWeightScore',
    'NormalizedScore',
    'NPScore',
    'SAScore',
    'ValidityScore',
]


def _read_pickle_gz(filename):
    with gzip.open(filename) as fin:
        data = pickle.load(fin)
    return data


# see https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py
def _readSAModel(filename):
    model_data = _read_pickle_gz(filename)
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    return outDict


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


class AbstractScore(object, metaclass=ABCMeta):

    @abstractmethod
    def compute(self, mol):
        """Compute score."""

    @property
    @abstractmethod
    def name(self):
        """Name of score."""

    @property
    @abstractmethod
    def min_score(self):
        """Return minimum score."""

    def map(self, iterable):
        return map(self.compute, iterable)


class NormalizedScore(AbstractScore):
    """Wrap scorer to return values in the range [0; 1]."""

    def __init__(self, scorer, max_val):
        self._scorer = scorer
        self._max_val = max_val

    @property
    def name(self):
        return "normalized_{}".format(self._scorer.name)

    @property
    def min_score(self):
        return 0.0

    def compute(self, mol):
        value = self._scorer.compute(mol)
        normed = remap(value, self._scorer.min_score, self._max_val)
        assert 0 <= normed <= 1, '{}={} not in [0; 1]'.format(self.name, normed)
        return normed


class CombinedScore(AbstractScore):
    def __init__(self, scorers):
        self._scorers = scorers

    @property
    def name(self):
        names = ['combined'] + [s.name for s in self._scorers]
        return "_".join(names)

    @property
    def min_score(self):
        amin = 1.0
        for s in self._scorers:
            amin *= s.min_score
        return amin

    def compute(self, mol):
        val = 1.0
        for s in self._scorers:
            val *= s.compute(mol)
        return val


class SAScore(AbstractScore):
    """Synthetic Accessibility Score

    A heuristic estimate of how hard (10) or how easy (1)
    it is to synthesize a given molecule.

    Source: https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
    """

    def __init__(self, sa_model_file):
        self._sa_model_file = sa_model_file
        self._sa_model = None

    @property
    def SA_model(self):
        if self._sa_model is None:
            self._sa_model = _readSAModel(self._sa_model_file)
        return self._sa_model

    @property
    def name(self):
        return 'synthetic_accessibility'

    @property
    def min_score(self):
        return 10.0

    def compute(self, mol):
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        sam = self.SA_model
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += sam.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore


class NPScore(AbstractScore):
    """Natural Product-likeness Score

    A numerical estimate that can be used to determine if
    a molecule is likely to be a natural product (0...5),
    a drug molecule (-3...3) or a synthetic product (-5...0).

    Source: https://github.com/rdkit/rdkit/tree/master/Contrib/NP_Score
    """

    def __init__(self, np_model_file):
        self._np_model_file = np_model_file
        self._np_model = None

    @property
    def name(self):
        return 'natural_product'

    @property
    def min_score(self):
        return -5  # not a hard lower bound

    @property
    def NA_model(self):
        if self._np_model is None:
            self._np_model = _read_pickle_gz(self._np_model_file)
        return self._np_model

    def compute(self, mol):
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()

        # calculating the score
        score = sum(self.NA_model.get(bit, 0) for bit in bits)
        score /= float(mol.GetNumAtoms())

        # preventing score explosion for exotic molecules
        if score > 4:
            score = 4. + math.log10(score - 4. + 1.)
        elif score < -4:
            score = -4. - math.log10(-4. - score + 1.)

        return score


class ValidityScore(AbstractScore):

    @staticmethod
    def to_valid_smiles(mol):
        s = Chem.MolToSmiles(mol, isomericSmiles=True)
        if mol is None:
            return None
        try:
            with disable_rdkit_log():
                Chem.SanitizeMol(mol)
        except ValueError:
            return None

        if ValidityScore.is_valid_smiles(s):
            return s

    @staticmethod
    def is_valid_smiles(s):
        is_valid = s != '' and '*' not in s and '.' not in s
        return is_valid

    @property
    def name(self):
        return 'validity'

    @property
    def min_score(self):
        return False

    def compute(self, mol):
        ret = ValidityScore.to_valid_smiles(mol)
        return ret is not None


class LogPScore(AbstractScore):
    """Water-octanol partition coefficient

    A property that measures how likely a molecule
    is able to mix with water. Hydrophilic compounds have
    a negative log partition, lipophilic compounds have
    a positive coefficient.
    """

    @property
    def name(self):
        return 'water_octanol_partition_coefficient'

    @property
    def min_score(self):
        raise NotImplementedError('minimum is undefined')

    def compute(self, mol):
        score = Chem.Crippen.MolLogP(mol)
        return score


class QEDScore(AbstractScore):
    """Quantitative Estimation of Drug-likeness (​QED)

    A 0 to 1 float value estimating how likely a molecule
    is a viable candidate for a drug.
    """

    @property
    def name(self):
        return 'quantitative_estimate_druglikeness'

    @property
    def min_score(self):
        return 0.0

    def compute(self, mol):
        return qed(mol)


class MolecularWeightScore(AbstractScore):
    """Computes molecular weight for given molecule."""

    @property
    def name(self):
        return 'molecular_weight'

    @property
    def min_score(self):
        return 0.0

    def compute(self, mol):
        return Descriptors.MolWt(mol)


class CycleLengthScore(AbstractScore):
    """Return the largest ring size with more than six atoms."""

    @property
    def name(self):
        return 'max_cycle_length'

    @property
    def min_score(self):
        return 0

    def compute(self, mol):
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max(map(len, cycle_list))

        return max(0, cycle_length - 6)


class PenalizedLogPScore(AbstractScore):
    """Penalized logP is the logP minus the synthetic accessibility (SA)
    score and the number of long cycles."""

    def __init__(self, sa_model_file, standardization_file=None):
        self._scorers = (
            LogPScore(),
            SAScore(sa_model_file),
            CycleLengthScore(),
        )
        if standardization_file is not None:
            with open(standardization_file, 'rb') as fin:
                self._norm = pickle.load(fin)
        else:
            self._norm = None

    @property
    def name(self):
        return 'penalized_logP'

    @property
    def min_score(self):
        if self._norm is None:
            raise NotImplementedError('minimum is undefined')

        return 0.0

    def compute(self, mol):
        scores = [s.compute(mol) for s in self._scorers]

        if self._norm is None:
            value = scores[0] - sum(scores[1:])
        else:
            self._standardize(scores)
            # make sure score is always within 0-1 interval
            value = np.clip(remap(scores[0] - sum(scores[1:]), -2, 1), 0.0, 1.0)
        return value

    def _standardize(self, scores):
        for i, (s, val) in enumerate(zip(self._scorers, scores)):
            scores[i] = remap(val, self._norm[s.name]['min'], self._norm[s.name]['max'])
