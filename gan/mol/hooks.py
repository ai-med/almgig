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
from os.path import join
from pprint import pformat
from typing import *
import numpy as np
from tensorflow.core.framework import summary_pb2
import tensorflow as tf

from .data import InputFunction


class FeedableTrainOpsHook(tf.train.SessionRunHook):

    def __init__(self, train_ops, train_steps,
                 input_fn: InputFunction,
                 return_feed_dict: bool = False) -> None:
        if not isinstance(train_ops, (list, tuple)):
            train_ops = [train_ops]
        self._train_ops = train_ops
        self._train_steps = train_steps
        self._input_fn = input_fn
        self._return_feed_dict = return_feed_dict

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            self._input_fn.get_next_batch()
            run_context.session.run(self._train_ops, feed_dict=self._input_fn.feed_dict)
        if self._return_feed_dict:
            gs = tf.train.get_global_step()
            return tf.train.SessionRunArgs(gs, feed_dict=self._input_fn.feed_dict)


class WithRewardTrainOpsHook(FeedableTrainOpsHook):

    def __init__(self,
                 train_ops,
                 train_steps,
                 input_fn: InputFunction,
                 mol_metrics) -> None:
        super(WithRewardTrainOpsHook, self).__init__(
            train_ops=train_ops,
            train_steps=train_steps,
            input_fn=input_fn)
        self._mol_metrics = mol_metrics
        self._pred_estimator = None

    def _get_rewards_for_batch(self, session):
        data = self._input_fn.data  # get reference to batch's data

        g = session.graph
        adj_tensor = g.get_tensor_by_name('Generator/discrete_adjacency:0')
        feat_tensor = g.get_tensor_by_name('Generator/discrete_features:0')

        adj_out, feat_out = session.run([adj_tensor, feat_tensor],
                                        feed_dict=self._input_fn.feed_dict)  # TODO would only need to feed embedding here

        reward_generated = self._mol_metrics.get_reward_metrics(zip(feat_out, adj_out))

        if tf.logging.get_verbosity() <= tf.logging.DEBUG:
            avg_real = np.mean(data['reward_real'])
            assert avg_real > 0, 'no rewards for real data.'
            tf.logging.debug('average reward: on real=%.4f, on generated: %.4f',
                             avg_real, np.mean(reward_generated))

        # update data for next batch in-place
        data['reward_generated'] = reward_generated

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            self._input_fn.get_next_batch()
            self._get_rewards_for_batch(run_context.session)
            run_context.session.run(self._train_ops, feed_dict=self._input_fn.feed_dict)

        gs = tf.train.get_global_step()
        return tf.train.SessionRunArgs(gs, feed_dict=self._input_fn.feed_dict)


MetricsFunc = Callable[[np.ndarray, np.ndarray], Dict[str, float]]


def _to_scalar_protobuf(name: str, value: Any) -> summary_pb2.Summary:
    value = float(value)
    buf = summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=name, simple_value=value)])
    return buf


def _merge_summaries(summaries: Iterable[summary_pb2.Summary]) -> summary_pb2.Summary:
    values = []
    for summ in summaries:
        values.extend(summ.value)
    return summary_pb2.Summary(value=values)


class EvalMoleculeMetricsHook(tf.train.SessionRunHook):

    def __init__(self,
                 metrics_fn: MetricsFunc,
                 adj_tensor: Union[str, tf.Tensor],
                 feat_tensor: Union[str, tf.Tensor],
                 output_dir: str) -> None:
        self._adj_tensor_or_name = adj_tensor
        self._feat_tensor_or_name = feat_tensor
        self._metrics_fn = metrics_fn
        self._output_dir = output_dir

    def _get_tensor(self, tensor_or_name: Union[str, tf.Tensor]) -> tf.Tensor:
        if isinstance(tensor_or_name, str):
            g = tf.get_default_graph()
            return g.get_tensor_by_name(tensor_or_name)
        return tensor_or_name

    def begin(self) -> None:
        self._adj_tensor = self._get_tensor(self._adj_tensor_or_name)
        self._feat_tensor = self._get_tensor(self._feat_tensor_or_name)
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._results = defaultdict(lambda: [])

    def before_run(self, run_context: tf.train.SessionRunContext) -> tf.train.SessionRunArgs:
        return tf.train.SessionRunArgs(fetches=[
            self._global_step_tensor, self._adj_tensor, self._feat_tensor])

    def after_run(self,
                  run_context: tf.train.SessionRunContext,
                  run_values: tf.train.SessionRunValues) -> None:
        global_step, adj_arr, feat_arr = run_values.results
        results = self._metrics_fn(feat_arr, adj_arr)

        for k, v in results.items():
            self._results[k].append(v)

        self._last_global_step = global_step

    def end(self, session):
        writer = tf.summary.FileWriterCache.get(self._output_dir)

        global_step = self._last_global_step
        msg = ['global_step = {}'.format(global_step)]
        for k, values in self._results.items():
            v = np.mean(values)
            msg.append('{} = {:.3f}'.format(k, v))
            buf = _to_scalar_protobuf('mol_metrics/{}'.format(k), v)
            writer.add_summary(buf, global_step=global_step)

        tf.logging.info(', '.join(msg))


class PredictAndEvalMolecule(tf.train.CheckpointSaverListener):

    def __init__(self, estimator, predict_fn, mol_metrics, output_dir):
        self.estimator = estimator
        self.predict_fn = predict_fn
        self.mol_metrics = mol_metrics
        self.output_dir = output_dir
        self._latest_path = None

    def after_save(self, session, global_step_value):
        # First call is due to CheckpointSaverHook.after_create_session,
        # which we are going to skip.
        # See tf.contrib.learn.experiment._EvalAndExportListener
        tf.logging.info("Checking for checkpoint in %s", self.output_dir)
        latest_path = tf.train.latest_checkpoint(self.output_dir)

        if not latest_path:
            tf.logging.warning("Skipping evaluation and export since model has not been "
                               "saved yet.")
        elif latest_path == self._latest_path:
            tf.logging.warning("Skipping evaluation due to same latest checkpoint %s.",
                               latest_path)
        else:
            self._latest_path = latest_path
            self._write_summary(global_step_value)

    def _write_summary(self, global_step_value):
        from .drawing import mols_to_image_summary

        mols = []
        rewards = []

        def mols_predictor():
            for pred in self.estimator.predict(self.predict_fn):
                feat, adj = pred['features'], pred['adjacency']
                mol = self.mol_metrics._as_molecule((feat, adj))
                if self.mol_metrics.is_valid(mol):
                    mols.append(mol)
                    reward = pred.get('reward')
                    if reward is not None:
                        rewards.append(pred['reward'])
                yield feat, adj

        results = self.mol_metrics.get_validation_metrics_summary(mols_predictor())

        predict_dir = join(self.output_dir, 'predict')
        writer = tf.summary.FileWriterCache.get(predict_dir)
        msg = ['global_step = {}'.format(global_step_value)]
        summaries = []
        for k, values in results.items():
            v = np.mean(values)
            msg.append('{} = {:.3f}'.format(k, v))
            buf = _to_scalar_protobuf('mol_metrics/{}'.format(k), v)
            summaries.append(buf)

        actual_rewards = self.mol_metrics.get_reward_metrics(mols)
        # sort molecules by reward (descending)
        mols = np.asarray(mols, dtype=object)
        o = np.argsort(-actual_rewards[:, 0])
        mols = mols[o]

        if len(rewards) > 0:
            assert len(mols) == len(rewards)

            mse = np.mean(np.square(actual_rewards - rewards))
            m = np.column_stack((actual_rewards, rewards))
            msg.append('reward_mse = {:.4f}'.format(mse))
            buf = _to_scalar_protobuf('mol_metrics/reward_mse', mse)
            summaries.append(buf)

        tf.logging.info(', '.join(msg))

        if len(mols) > 0:
            summary = mols_to_image_summary(mols,
                                            name='generated',
                                            num_cols=8,
                                            sub_image_shape=(125, 125))
            summaries.append(summary)

        writer.add_summary(_merge_summaries(summaries), global_step=global_step_value)


class RestoreFromCheckpointHook(tf.train.SessionRunHook):

    def __init__(self, filename: str) -> None:
        self.filename = filename

        self._saver = None

    def _create_saver(self):
        reader = tf.train.NewCheckpointReader(self.filename)
        ckpt_tensor_names = set(reader.get_variable_to_shape_map())

        def _tensor_name(op):
            return op.name.split(':', 1)[0]

        model_tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        model_tensor_names = set(map(_tensor_name, model_tensors))

        in_ckpt_tensors = model_tensor_names.intersection(ckpt_tensor_names)
        if len(in_ckpt_tensors) != len(model_tensor_names):
            tf.logging.warning('Only found %d of %d variables in checkpoint %s',
                               len(in_ckpt_tensors),
                               len(model_tensor_names),
                               self.filename)
            diff = model_tensor_names - in_ckpt_tensors
            tf.logging.warning('The following %d variables are not restored:\n%s',
                               len(diff), pformat(diff))

        restore_vars = [op for op in model_tensors if _tensor_name(op) in in_ckpt_tensors]
        assert len(restore_vars) == len(in_ckpt_tensors)
        return tf.train.Saver(restore_vars)

    def begin(self):
        self._saver = self._create_saver()

    def after_create_session(self, session, coord):
        self._saver.restore(session, self.filename)
