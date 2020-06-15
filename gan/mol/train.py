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
from argparse import Namespace
import datetime
import logging
from pathlib import Path
import pickle
from pprint import pformat
import time
from typing import Any, Iterable, Optional, Tuple
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorpack.dataflow import RepeatedData, PrefetchDataZMQ

from ..utils import PrintParameterSummary
from .base import MolGANTrainOps
from .data import create_dataflow
from .data.graph2mol import get_decoder, get_dataset, MoleculeData
from .estimator import EstimatorTrainHooks
from .experiments import Experiment
from .hooks import FeedableTrainOpsHook, PredictAndEvalMolecule, RestoreFromCheckpointHook, WithRewardTrainOpsHook
from .metrics import GraphMolecularMetrics
from .molgan import MolGANParameters
from .validate import validate

LOG = logging.getLogger(__name__)


class ScheduledHyperParamSetter(tf.train.SessionRunHook):

    def __init__(self,
                 name: str,
                 initial_value: float,
                 schedule: Iterable[Tuple[int, float]],
                 steps_per_epoch: int,
                 interpolate: bool = False) -> None:
        self.name = name
        self.initial_value = initial_value
        self.schedule = sorted(schedule, key=lambda x: int(x[0]))
        self.steps_per_epoch = steps_per_epoch
        self.interpolate = interpolate

        self._last_value = initial_value
        self._last_epoch = 0

    def _next_value(self, epoch: int) -> Tuple[Optional[int], Optional[float]]:
        e = None
        v = None
        for e, v in self.schedule:
            if e >= epoch:
                break
        return e, v

    def get_value_to_set(self, epoch: int) -> Optional[float]:
        next_epoch, next_value = self._next_value(epoch)
        if next_epoch == epoch:
            self._last_value = next_value
            self._last_epoch = next_epoch
            return next_value

        if self.interpolate and epoch < next_epoch:
            m = (next_value - self._last_value) / (next_epoch - self._last_epoch)
            value = m * (epoch - self._last_epoch) + self._last_value
            return value

    def after_create_session(self, session, coord):
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.name:
                self._var = v
                break
        else:
            raise ValueError("{} is not a variable in the graph!".format(self.name))

        self._global_step_tensor = tf.train.get_global_step()

    def before_run(self, run_context):
        sess = run_context.session
        next_gs = tf.train.global_step(sess, self._global_step_tensor) + 1
        next_epoch = next_gs // self.steps_per_epoch

        value = self.get_value_to_set(next_epoch)

        if value is not None:
            current_value = self._var.eval(sess)
            if current_value != value:
                LOG.info('epoch: %d, setting hyper-parameter %s: %s -> %s',
                        next_epoch, self.name, current_value, value)

                self._var.load(value, sess)


def train_and_evaluate(args: Namespace,
                       experiment: Experiment) -> None:
    LOG.info('Parameters:\n%s', pformat(vars(args)))
    with (args.model_dir / 'args.pkl').open('wb') as fp:
        pickle.dump(args, fp)

    data_params = get_dataset(args.dataset)
    norm_file = args.data_dir / 'norm_penLogP.pkl'

    mol_metrics = GraphMolecularMetrics(
        get_decoder(args.dataset, strict=True),
        args.reward_type,
        norm_file)

    if args.epochs > 0:
        train(args, data_params, experiment, mol_metrics)
    else:
        LOG.warning('Skipping training')
    evaluate(args, data_params, experiment, mol_metrics)


def train(args: Namespace,
          data_params: MoleculeData,
          experiment: Experiment,
          mol_metrics: GraphMolecularMetrics) -> None:
    ds_train = create_dataflow(args.data_dir, 'train', args.batch_size)

    ds_train_repeat = PrefetchDataZMQ(ds_train, nr_proc=1)
    # times 2, because we consume 2 batches per step
    ds_train_repeat = RepeatedData(ds_train_repeat, 2 * args.epochs)

    train_input_fn = experiment.make_train_fn(
        ds_train_repeat, args.batch_size, args.num_latent,
        data_params)

    def hooks_fn(train_ops: MolGANTrainOps,
                 train_steps: tfgan.GANTrainSteps) -> EstimatorTrainHooks:
        if train_ops.valuenet_train_op is not None:
            generator_hook = FeedableTrainOpsHook(
                train_ops.generator_train_op,
                train_steps.generator_train_steps,
                train_input_fn,
                return_feed_dict=False)

            discriminator_hook = WithRewardTrainOpsHook(
                [train_ops.discriminator_train_op, train_ops.valuenet_train_op],
                train_steps.discriminator_train_steps,
                train_input_fn,
                mol_metrics)
        else:
            generator_hook = FeedableTrainOpsHook(
                train_ops.generator_train_op,
                train_steps.generator_train_steps,
                train_input_fn,
                return_feed_dict=True)

            discriminator_hook = FeedableTrainOpsHook(
                train_ops.discriminator_train_op,
                train_steps.discriminator_train_steps,
                train_input_fn)
        return [generator_hook, discriminator_hook]

    model = experiment.make_model_fn(args, data_params, hooks_fn)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # enable XLA JIT
    # sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    config = tf.estimator.RunConfig(
        model_dir=str(args.model_dir),
        session_config=sess_config,
        save_summary_steps=ds_train.size(),
        save_checkpoints_secs=None,
        save_checkpoints_steps=4 * ds_train.size(),
        keep_checkpoint_max=2)

    estimator = tf.estimator.Estimator(model.model_fn, config=config)

    train_hooks = [PrintParameterSummary()]
    if args.restore_from_checkpoint is not None:
        train_hooks.append(
            RestoreFromCheckpointHook(str(args.restore_from_checkpoint))
        )

    if args.debug:
        from tensorflow.python import debug as tf_debug

        train_hooks.append(tf_debug.TensorBoardDebugHook("localhost:6064"))

    predict_fn = experiment.make_predict_fn(args.data_dir, args.num_latent,
                                            n_samples=1000, batch_size=1000)
    ckpt_listener = PredictAndEvalMolecule(estimator, predict_fn, mol_metrics,
                                           str(args.model_dir))

    hparams_setter = [
        ScheduledHyperParamSetter(
            'generator_learning_rate:0',
            args.generator_learning_rate,
            [(80, 0.5 * args.generator_learning_rate),
             (150, 0.1 * args.generator_learning_rate),
             (200, 0.01 * args.generator_learning_rate)],
             steps_per_epoch=ds_train.size()),
        ScheduledHyperParamSetter(
            'discriminator_learning_rate:0',
            args.discriminator_learning_rate,
            [(80, 0.5 * args.discriminator_learning_rate),
             (150, 0.1 * args.discriminator_learning_rate),
             (200, 0.01 * args.discriminator_learning_rate)],
             steps_per_epoch=ds_train.size())
    ]
    train_hooks.extend(hparams_setter)

    if args.weight_reward_loss > 0:
        if args.weight_reward_loss_schedule == 'linear':
            lambda_setter = ScheduledHyperParamSetter(
                model.params, 'lam',
                [(args.reward_loss_delay, 1.0), (args.epochs, 1.0 - args.weight_reward_loss)],
                True)
        elif args.weight_reward_loss_schedule == 'const':
            lambda_setter = ScheduledHyperParamSetter(
                model.params, 'lam',
                [(args.reward_loss_delay + 1, 1.0 - args.weight_reward_loss)],
                False)
        else:
            raise ValueError('unknown schedule: {!r}'.format(args.weight_reward_loss_schedule))

        hparams_setter.append(lambda_setter)

    train_start = time.time()
    estimator.train(train_input_fn, hooks=train_hooks,
                    saving_listeners=[ckpt_listener])
    train_end = time.time()

    time_d = datetime.timedelta(seconds=int(train_end - train_start))
    LOG.info('Training for %d epochs finished in %s', args.epochs, time_d)


def evaluate(args: Namespace,
             data_params: MoleculeData,
             experiment: Experiment,
             mol_metrics: GraphMolecularMetrics) -> None:
    model = experiment.make_model_fn(args, data_params, hooks_fn=None)

    config = tf.estimator.RunConfig(
        model_dir=str(args.model_dir),
        save_summary_steps=None,
        save_checkpoints_secs=None,
        save_checkpoints_steps=None)

    estimator = tf.estimator.Estimator(model.model_fn, config=config)

    def _gen_predict_fn(n_samples, batch_size):
        return experiment.make_predict_fn(data_dir=args.data_dir,
                                          num_latent=args.num_latent,
                                          n_samples=n_samples,
                                          batch_size=batch_size,
                                          seed=None)

    cpkt_path = Path(estimator.latest_checkpoint())

    for v in ('train', 'test'):
        smiles_file = str(args.data_dir / f'{v}.smiles')
        json_file = 'distribution-learning_' + cpkt_path.name + '.json'
        if v == 'test':
            json_file = 'test_' + json_file
        output_stats = cpkt_path.with_name(json_file)
        validate(estimator, _gen_predict_fn, mol_metrics.conv,
                 smiles_file, output_stats)

    if hasattr(model, 'encoder_fn'):
        from .alice.predict import EmbeddingSaver

        emb_dir = args.model_dir / 'embedding'
        eval_fn = experiment.make_predict_fn(data_dir=args.data_dir,
                                             num_latent=args.num_latent,
                                             n_samples=1024,
                                             batch_size=1024)
        emb_saver = EmbeddingSaver(emb_dir, args.reward_type, mol_metrics.conv)
        emb_saver.save_as_checkpoint(estimator, eval_fn)
