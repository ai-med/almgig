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
from contextlib import contextmanager
from typing import Dict, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit import rdBase


@contextmanager
def disable_rdkit_log():
    rdBase.DisableLog('rdApp.error')
    try:
        yield
    finally:
        rdBase.EnableLog('rdApp.error')


def onehot_to_dense(node_labels: np.ndarray,
                    edge_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if edge_labels.ndim == 3 and node_labels.ndim == 2:
        batch_offset = 0
    elif edge_labels.ndim == 4 and node_labels.ndim == 3:
        batch_offset = 1
    else:
        raise ValueError('dimensions mismatch')

    idx_edge_i = batch_offset + 1
    idx_edge_j = batch_offset + 2

    assert edge_labels.shape[idx_edge_i] == edge_labels.shape[idx_edge_j], \
        '{} != {}'.format(edge_labels.shape[idx_edge_i], edge_labels.shape[idx_edge_j])
    assert node_labels.shape[batch_offset] == edge_labels.shape[idx_edge_i], \
        '{} != {}'.format(node_labels.shape[batch_offset], edge_labels.shape[idx_edge_i])
    edge_labels = np.argmax(edge_labels, axis=batch_offset)
    node_labels = np.argmax(node_labels, axis=-1)
    return node_labels, edge_labels


class MoleculeData(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self._bond_decoder = None
        self._atom_decoder = None

    @property
    @abstractmethod
    def supported_atoms(self):
        """Return list of supported atoms"""

    @property
    def valence_of_supported_atoms(self):
        valence = [0] + [Chem.MolFromSmiles(a).GetAtomWithIdx(0).GetTotalValence()
                         for a in self.supported_atoms]
        valence = np.array(valence, dtype=np.float32)
        return valence

    def get_atom_decoder(self):
        if self._atom_decoder is None:
            self._atom_decoder = MoleculeData._create_atom_decoder(self.supported_atoms)
        return self._atom_decoder

    @staticmethod
    def _create_atom_decoder(atom_symbols):
        decoder = {0: 0}
        for i, val in enumerate(atom_symbols):
            decoder[i + 1] = val
        return decoder

    def get_bond_decoder(self):
        if self._bond_decoder is None:
            self._bond_decoder = MoleculeData._create_bond_decoder()
        return self._bond_decoder

    @staticmethod
    def _create_bond_decoder():
        decoder = {0: Chem.rdchem.BondType.ZERO,
                   1: Chem.rdchem.BondType.SINGLE,
                   2: Chem.rdchem.BondType.DOUBLE,
                   3: Chem.rdchem.BondType.TRIPLE}
        return decoder


class GDB9(MoleculeData):
    MAX_NODES = 9
    NUM_NODE_TYPES = 4
    NUM_EDGE_TYPES = 3

    def __init__(self):
        super(GDB9, self).__init__()

    @property
    def supported_atoms(self):
        return 'C', 'F', 'N', 'O'


class ZINC(MoleculeData):
    MAX_NODES = 37
    NUM_NODE_TYPES = 9
    NUM_EDGE_TYPES = 3

    def __init__(self):
        super(ZINC, self).__init__()

    @property
    def supported_atoms(self):
        return 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S'


class Graph2Mol:

    def __init__(self,
                 atom_decoder: Dict[int, Union[str, int]],
                 bond_decoder: Dict[int, int],
                 strict: bool = False) -> None:
        self.atom_decoder = atom_decoder
        self.bond_decoder = bond_decoder
        self.strict = strict

    def to_mol(self, node_labels: np.ndarray, edge_labels: np.ndarray) -> Chem.rdchem.Mol:
        if edge_labels.ndim == 3 and node_labels.ndim == 2:
            # reverse one-hot encoding
            node_labels, edge_labels = onehot_to_dense(node_labels, edge_labels)
        elif edge_labels.ndim == 2 and node_labels.ndim == 1:
            assert edge_labels.shape[0] == edge_labels.shape[1]
            assert node_labels.shape[0] == edge_labels.shape[0]
        else:
            raise ValueError(
                'unsupported dimensions: node_labels.ndim = {}, edge_labels.ndim = {}'.format(
                    node_labels.ndim, edge_labels.ndim))

        mol = Chem.RWMol()
        idx_map = {}
        for i, node_label in enumerate(node_labels):
            if node_label != 0:  # skip "not-a-atom" type
                idx = mol.AddAtom(Chem.Atom(self.atom_decoder[node_label]))
                idx_map[i] = idx

        rows, cols = np.nonzero(np.triu(edge_labels, k=1))
        for start, end in zip(rows, cols):
            node_start = int(start)
            node_end = int(end)
            atom_start = idx_map.get(node_start)
            atom_end = idx_map.get(node_end)
            bond_type = self.bond_decoder[edge_labels[node_start, node_end]]
            if atom_start is None or atom_end is None:
                # only absent bond type is allowed
                if bond_type == Chem.rdchem.BondType.ZERO:
                    continue
                raise ValueError('cannot have a bond to a non-atom type')

            mol.AddBond(atom_start, atom_end, bond_type)

        mol = Chem.Mol(mol)
        assert (node_labels != 0).sum() == mol.GetNumAtoms(), \
            'expected {} atoms, but got {}'.format((node_labels != 0).sum(), mol.GetNumAtoms())
        assert len(rows) == mol.GetNumBonds(), \
            'expected {} bonds, but got {}'.format(len(rows), mol.GetNumBonds())

        if self.strict:
            with disable_rdkit_log():
                # Sometimes, SanitizeMol creates an invalid molecule,
                # for which the second call will throw an exception.
                Chem.SanitizeMol(mol)
                Chem.SanitizeMol(mol)

        return mol


def get_dataset(name: str) -> MoleculeData:
    if name == 'gdb9':
        dec_cls = GDB9
    elif name == 'zinc':
        dec_cls = ZINC
    else:
        raise ValueError('{!r} is not supported'.format(name))
    return dec_cls()


def get_decoder(name: str, strict: bool = False) -> Graph2Mol:
    dec = get_dataset(name)
    return Graph2Mol(dec.get_atom_decoder(), dec.get_bond_decoder(), strict)
