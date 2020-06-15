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
import logging
import sys

from nevae import train, predict, sample
from nevae.utils import setup_logging

LOG = logging.getLogger('nevae')


def main():
    setup_logging(logging.INFO)

    args = sys.argv[1:]

    fname = 'data/gdb9-with_hydrogen/gdb9_hydrogen_train.pkl'

    nodes = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
    edges = (12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25, 26)
    for n_nodes, max_edges in zip(nodes, edges):
        argv_train = args + [
            '-f', fname,
            '--n_nodes', str(n_nodes),
            '--max_edges', str(max_edges),
        ]

        LOG.info("Calling train with arguments:\n%s\n", argv_train)
        train.main(args=argv_train)

        args_train, _ = train.create_parser().parse_known_args(argv_train)
        model_dir = args_train.model_dir

        argv_predict = [
            '--model_dir', str(model_dir),
            '--n_nodes', str(n_nodes),
            '--num_latent', str(args_train.num_latent),
            '--num_samples', '50',
            '--output', f'{model_dir}/predictions_{n_nodes}.pkl',
        ]
        LOG.info("Calling predict with arguments:\n%s\n", argv_predict)
        predict.main(args=argv_predict)

        argv_sample = [
            '--input', f'{model_dir}/predictions_{n_nodes}.pkl',
            '--output', f'{model_dir}/predictions_{n_nodes}.csv',
        ]
        LOG.info("Calling sample with arguments:\n%s\n", argv_sample)
        sample.main(args=argv_sample)


if __name__ == '__main__':
    main()
