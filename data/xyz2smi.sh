#!/bin/bash
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
set -ue

for afile in $@
do
    mol_smi=$(tail -n 2 "${afile}" | head -n 1 | cut -f1)
    mol_id=$(head -n 2 "${afile}" | tail -n 1 | cut -f1 | cut -d" " -f2)
    mol_name=$(printf "gdb_%06d" "${mol_id}")

    echo -e "${mol_smi}\t${mol_name}"
done
