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
import io
from typing import Collection, Tuple
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tensorflow.core.framework import summary_pb2

Shape = Tuple[int, int]


def mol_to_image_file(mol: Chem.rdchem.Mol,
                      filename: str,
                      size: Shape = (300, 300)) -> None:
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, filename, size=size)


def mol_to_image(mol: Chem.rdchem.Mol,
                 size: Shape = (300, 300)) -> Image.Image:
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    return img


def mols_to_image_summary(mols: Collection[Chem.rdchem.Mol],
                          name: str,
                          num_cols: int = 3,
                          sub_image_shape: Shape = (200, 200)) -> summary_pb2.Summary:
    for m in mols:
        AllChem.Compute2DCoords(m)
    image = Draw.MolsToGridImage(mols, molsPerRow=num_cols, subImgSize=sub_image_shape)

    with io.BytesIO() as output:
        image.save(output, format='PNG')
        image_string = output.getvalue()

    summary = summary_pb2.Summary.Image(
        height=image.height,
        width=image.width,
        colorspace=len(image.getbands()),
        encoded_image_string=image_string)
    summary = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=name, image=summary)])

    return summary
