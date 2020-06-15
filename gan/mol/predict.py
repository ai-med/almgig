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
from pathlib import Path
from typing import Tuple, Union
import pickle
import tensorflow as tf

from .data.graph2mol import get_dataset

LOG = logging.getLogger(__name__)


def load_estimator_from_dir(model_dir: Union[str, Path],
                            return_num_latent: bool = False,
                            **kwargs) -> Union[tf.estimator.Estimator, Tuple[tf.estimator.Estimator, int]]:
    model_dir = Path(model_dir)
    with (model_dir / 'args.pkl').open('rb') as fp:
        args = pickle.load(fp)
        args.model_dir = model_dir

    for k, v in kwargs.items():
        LOG.info('Overriding option %s: %r -> %r', k, getattr(args, k), v)
        setattr(args, k, v)

    setattr(args, "without_gated_gcn", False)

    if not (args.model_dir / 'checkpoint').exists():
        raise ValueError(f'no checkpoints in {args.model_dir}')

    experiment = args.experiment_cls()

    config = tf.estimator.RunConfig(
        model_dir=str(args.model_dir),
        save_summary_steps=None,
        save_checkpoints_secs=None,
        save_checkpoints_steps=None)

    data_params = get_dataset(args.dataset)

    model = experiment.make_model_fn(args, data_params, hooks_fn=None)
    estimator = tf.estimator.Estimator(model.model_fn, config=config)
    LOG.info('Loading model %s from %s',
             model.__class__.__name__,
             estimator.latest_checkpoint())
    if return_num_latent:
        return estimator, model.params.num_latent
    return estimator
