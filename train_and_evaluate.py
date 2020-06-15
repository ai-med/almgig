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
import logging
from pathlib import Path
from gan.mol.metrics import RewardType
from gan.mol.train import train_and_evaluate
from gan.mol.experiments import AliceExperiment, NonSaturatingGanExperiment, WGanExperiment


def _create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    gan_cli = subparsers.add_parser('gan')
    gan_cli.set_defaults(experiment_cls=NonSaturatingGanExperiment)
    _add_arguments(gan_cli)

    wgan_cli = subparsers.add_parser('wgan')
    wgan_cli.set_defaults(experiment_cls=WGanExperiment)
    _add_arguments(wgan_cli)

    almgig_cli = subparsers.add_parser('almgig')
    almgig_cli.set_defaults(experiment_cls=AliceExperiment)
    _add_arguments(almgig_cli)
    return parser


def _add_arguments(parser):
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=['gdb9', 'zinc'], required=True)
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--restore_from_checkpoint', type=Path)
    parser.add_argument('--debug', action='store_true', default=False)

    group = parser.add_argument_group('Training')
    group.add_argument('--batch_size', type=int, default=128)
    group.add_argument('--epochs', type=int, default=10)
    group.add_argument('--summarize_gradients', action='store_true', default=False)

    group = parser.add_argument_group('Optimizer')
    group.add_argument('--generator_learning_rate', type=float, default=0.001)
    group.add_argument('--discriminator_learning_rate', type=float, default=0.003)
    group.add_argument('--beta1', type=float, default=0.9)
    group.add_argument('--beta2', type=float, default=0.999)

    group = parser.add_argument_group('Architecture')
    group.add_argument('--num_latent', type=int, default=8)
    group.add_argument('--temperature', type=float, default=1.0)

    group = parser.add_argument_group('Reinforcement learning')
    group.add_argument('--reward_loss_delay', type=int, default=3)
    group.add_argument('--weight_reward_loss_schedule', choices=['const', 'linear'], default='const')
    group.add_argument('--weight_reward_loss', type=float, default=0.0)
    group.add_argument('--reward_type', type=RewardType.from_string,
                       metavar=RewardType.metavar(),
                       required=True)

    group = parser.add_argument_group('Regularization')
    group.add_argument('--weight_gradient_penalty', type=float, default=10.0)
    group.add_argument('--connectivity_penalty_weight', type=float, default=0.01)
    group.add_argument('--valence_penalty_weight', type=float, default=0.5)
    group.add_argument('--variance_penalty_weight', type=float, default=-0.5)

    group = parser.add_argument_group('Discriminator')
    group.add_argument('--without_cycle_discriminator', action='store_true', default=False)
    group.add_argument('--without_unary_discriminator', action='store_true', default=False)
    group.add_argument('--without_gcn_skip_connections', action='store_true', default=False)
    group.add_argument('--without_gated_gcn', action='store_true', default=False)

    group = parser.add_argument_group('Generator')
    group.add_argument('--without_generator_skip_connections', action='store_true', default=False)

    return parser


def config_logger(model_dir):
    from guacamol.utils.data import get_time_string

    timestring = get_time_string()
    fh = logging.FileHandler(model_dir / '{}-train.log'.format(timestring))
    fh.setFormatter(logging.Formatter(logging.BASIC_FORMAT))

    # configure root logger
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(sh)

    for name in ('gan', 'guacamol', 'tensorflow', 'tensorpack'):
        logger = logging.getLogger(name)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        if name == 'tensorflow':
            # avoid double logging
            logger.propagate = False

    fh.setLevel(logging.DEBUG)


def main(args=None):
    parser = _create_parser()
    args = parser.parse_args(args=args)

    if not hasattr(args, 'experiment_cls'):
        parser.error('You need to specify the model to train.')

    if not args.model_dir.exists():
        args.model_dir.mkdir(parents=True)

    config_logger(args.model_dir)

    train_and_evaluate(args, args.experiment_cls())


if __name__ == '__main__':
    main()
