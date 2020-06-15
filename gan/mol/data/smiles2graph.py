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
from itertools import accumulate
import logging
from pathlib import Path
import pickle
import platform
from typing import Any, Collection, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import openbabel
import pybel
from rdkit import Chem
from tqdm import tqdm

from guacamol.utils.chemistry import filter_and_canonicalize, \
    initialise_neutralisation_reactions, split_charged_mol
from guacamol.utils.data import get_time_string

LOG = logging.getLogger(__name__)


class Mol2Graph:

    def __init__(self, add_hydrogens: bool = False) -> None:
        self._add_hydrogens = add_hydrogens
        self._edge_types = set()
        self._node_types = set()
        self._element_tabel = openbabel.OBElementTable()

    @property
    def sorted_edge_types(self) -> List[int]:
        return sorted(list(self._edge_types))

    @property
    def sorted_node_types(self) -> List[str]:
        return sorted(list(self._node_types))

    def to_graph(self, molw: pybel.Molecule) -> nx.Graph:
        mol = molw.OBMol
        if self._add_hydrogens:
            mol.AddHydrogens()
        else:
            mol.DeleteHydrogens()

        g = nx.Graph()
        for atom in openbabel.OBMolAtomIter(mol):
            node_features = {'Symbol': self._element_tabel.GetSymbol(atom.GetAtomicNum())}
            g.add_node(atom.GetIdx(), **node_features)
            self._node_types.add(node_features['Symbol'])

        for b in openbabel.OBMolBondIter(mol):
            idx_a = b.GetBeginAtomIdx()
            idx_b = b.GetEndAtomIdx()
            edge_features = {'BondType': b.GetBondOrder()}
            g.add_edge(idx_a, idx_b, **edge_features)
            self._edge_types.add(edge_features['BondType'])

        # canonical SMILES format
        text = molw.write(format='can')
        g.graph['smiles'] = text.split('\t', 1)[0]
        g.graph['name'] = molw.title.split('\t', 1)[0]

        return g


class BatchConverter:

    def __init__(self) -> None:
        self._conv = None

    def _read_and_convert(self, mols: Iterable[pybel.Molecule]) -> List[nx.Graph]:
        data = []

        for m in tqdm(mols):
            g = self._conv.to_graph(m)
            if g.number_of_nodes() <= 1:
                LOG.warning('%s has %d atoms. Skipping it.', g.graph['name'], g.number_of_nodes())
                continue
            data.append(g)

        return data

    def _encode_atom_and_bond_types(self, data: List[nx.Graph]) -> None:
        edge_types = self._conv.sorted_edge_types
        node_types = self._conv.sorted_node_types

        print('Found %d edge types: %r' % (len(edge_types), edge_types))
        print('Found %d node types: %r' % (len(node_types), node_types))

        for g in data:
            for u, v, d in g.edges(data=True):
                i = edge_types.index(d['BondType'])
                g.get_edge_data(u, v)['BondTypeCode'] = i + 1
            for anode, d in g.nodes(data=True):
                i = node_types.index(d['Symbol'])
                g.nodes[anode]['AtomCode'] = i + 1

    def convert_all(self, mols: Iterable[pybel.Molecule], add_hydrogens: bool = False) -> List[nx.Graph]:
        self._conv = Mol2Graph(add_hydrogens=add_hydrogens)
        data = self._read_and_convert(mols)
        self._encode_atom_and_bond_types(data)
        return data


def get_argparser():
    timestring = get_time_string()
    parser = argparse.ArgumentParser(description='Data Preparation for GuacaMol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--destination', default='.', help='Download and Output location')
    parser.add_argument('-i', '--input', required=True, help='Filename of input smiles file')
    parser.add_argument('--output_prefix', default=timestring, help='Prefix of the output file')
    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cores to use')
    parser.add_argument('--seed', default=9325, type=int, help='Random number seed')
    parser.add_argument('--with_hydrogens', action='store_true', default=False,
                        help='Whether to add hydrogen nodes to the graph.')
    return parser


class AllowedSmilesCharDictionary(object):
    """
    A fixed dictionary for druglike SMILES.
    """

    def __init__(self) -> None:
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

    def allowed(self, smiles: str) -> bool:
        """
        Determine if SMILES string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                LOG.warning('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True


def get_raw_smiles(file_name, smiles_char_dict, open_fn) -> List[Tuple[str, str]]:
    """
    Extracts the raw smiles from an input file.
    open_fn will open the file to iterate over it (e.g. use open_fn=open or open_fn=filegzip.open)
    extract_fn specifies how to process the lines, choose from
    Pre-filter molecules of 5 <= length <= 200, because processing larger molecules (e.g. peptides) takes very long.

    Returns:
       a list of SMILES strings
    """
    data = []
    # open the gzipped chembl filegzip.open
    with open_fn(file_name, 'rt') as f:

        line_count = 0
        for line in f:

            line_count += 1
            # extract the canonical smiles column
            if platform.system() == "Windows":
                line = line.decode("utf-8")

            values = line.rstrip().split('\t', maxsplit=1)
            if len(values) == 1:
                smiles = values[0]
                name = f'line_{line_count}'
            else:
                smiles, name = values

            # only keep reasonably sized molecules
            if 5 <= len(smiles) <= 200:

                smiles = split_charged_mol(smiles)

                if smiles_char_dict.allowed(smiles):
                    # check whether the molecular graph consists of
                    # multiple connected components (eg. in salts)
                    # if so, just keep the largest one

                    data.append((smiles, name))

        LOG.info('Processed %d lines.', line_count)

    return data


def write_smiles(dataset: Iterable[str], filename: str):
    """
    Dumps a list of SMILES into a file, one per line
    """
    n_lines = 0
    with open(filename, 'w') as out:
        for smiles_str in dataset:
            out.write('%s\n' % smiles_str)
            n_lines += 1
    LOG.info('%s contains %d molecules', filename, n_lines)


def random_split(data: Collection[np.ndarray],
                 lengths: Collection[int],
                 random_state: Union[int, np.random.RandomState, None]) -> List[Any]:
    """Randomly split a dataset into non-overlapping new datasets of given lengths.

    Parameters
    ----------
    data : tuple
        A list of array-like objects to split.
    lengths : sequence
        Lengths of splits to be produced
    random_state : int
        Random number seed.

    Returns
    -------
    splits : list
    """
    n = len(data[0])
    for v in data[1:]:
        if len(v) != n:
            raise ValueError("all data items must have the same length")

    if sum(lengths) != n:
        raise ValueError("Sum of input lengths does not equal the length of the input graphs")

    if isinstance(random_state, int):
        rnd = np.random.RandomState(random_state)
    else:
        rnd = random_state
    indices = np.arange(n, dtype=np.intp)
    rnd.shuffle(indices)

    outputs = []
    for offset, length in zip(accumulate(lengths), lengths):
        ind = indices[offset - length:offset]
        item = []
        for d in data:
            item.append(d[ind])
        outputs.append(tuple(item))

    return outputs


def filter_and_canonicalize_with_name(smiles_str: str,
                                      name: str,
                                      *args,
                                      **kwargs) -> Tuple[str, str]:
    smi = filter_and_canonicalize(smiles_str, *args, **kwargs)
    smi = smi[0]
    mol = Chem.MolFromSmiles(smi)
    assert smi == Chem.MolToSmiles(mol, isomericSmiles=True)

    # drop molecules with charge
    if has_charge(mol):
        smi = None

    return smi, name


def has_charge(mol: Chem.rdchem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return True
    return False


def ob_mol_from_smiles(smi: str) -> Tuple[pybel.Molecule, str]:
    mol = pybel.readstring('smi', smi)
    smi_ob, _ = mol.write(format='can').split('\t', 1)
    return mol, smi_ob


def main():
    """Get data.

    Preprocessing steps:

    1) filter SMILES shorter than 5 and longer than 200 chars and those with forbidden symbols
    2) canonicalize, neutralize, only permit smiles shorter than 100 chars
    3) shuffle and write files.
    """
    argparser = get_argparser()
    args = argparser.parse_args()

    # Set constants
    rnd = np.random.RandomState(args.seed)
    neutralization_rxns = initialise_neutralisation_reactions()
    smiles_dict = AllowedSmilesCharDictionary()

    LOG.info('Preprocessing molecules...')

    raw_smiles = get_raw_smiles(args.input, smiles_char_dict=smiles_dict, open_fn=open)
    tanimoto_cutoff = 100  # effectively no cutoff
    holdout_set = set([])
    holdout_fps = []
    file_prefix = args.output_prefix

    LOG.info('Standardizing %d molecules using %d cores...', len(raw_smiles), args.n_jobs)

    # Process all the SMILES in parallel
    runner = Parallel(n_jobs=args.n_jobs, verbose=2)

    joblist = (delayed(filter_and_canonicalize_with_name)(smiles_str,
                                                          name,
                                                          holdout_set=holdout_set,
                                                          holdout_fps=holdout_fps,
                                                          neutralization_rxns=neutralization_rxns,
                                                          tanimoto_cutoff=tanimoto_cutoff,
                                                          include_stereocenters=False)
               for smiles_str, name in raw_smiles)

    output = runner(joblist)

    # Put all nonzero molecules in a list, remove duplicates, and sort
    all_good_smiles = []
    all_good_mols = []
    all_good_names = []
    unique_items = set()
    for smi, name in output:
        if smi is not None and smi not in unique_items:
            mol, smi_ob = ob_mol_from_smiles(smi)
            mol.title = name
            all_good_smiles.append(smi_ob)
            all_good_mols.append(mol)
            all_good_names.append(name)
            unique_items.add(smi)

    all_good_smiles = np.array(all_good_smiles)
    all_good_names = np.array(all_good_names)
    all_good_mols = np.array(all_good_mols, dtype=object)

    o = np.argsort(all_good_smiles)
    all_good_smiles = all_good_smiles[o]
    all_good_names = all_good_names[o]
    all_good_mols = all_good_mols[o]

    LOG.info('Ended up with %d molecules.', len(all_good_smiles))

    # Split into train-dev-test
    lengths = [int(v * len(all_good_smiles)) for v in [0.8, 0.1]]
    lengths.append(len(all_good_smiles) - sum(lengths))
    LOG.info('Splitting data into %d subsets: %s', len(lengths), lengths)

    split_data = random_split((all_good_smiles, all_good_names, all_good_mols), lengths, random_state=rnd)

    def paste(x):
        a, b = x
        return "{}\t{}".format(a, b)

    dest = Path(args.destination)
    for name, (values, names, mols) in zip(['train', 'valid', 'test'], split_data):
        assert len(values) == len(names)
        assert len(values) == len(mols)

        path = dest / f'{file_prefix}_{name}.smiles'
        write_smiles(map(paste, zip(values, names)), path)

        pkl_file = path.with_suffix('.pkl')
        data = BatchConverter().convert_all(mols, add_hydrogens=args.with_hydrogens)
        LOG.info('Writing graphs to %s', pkl_file)
        with pkl_file.open('wb') as fout:
            pickle.dump(data, fout)

    print('You are ready to go.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
