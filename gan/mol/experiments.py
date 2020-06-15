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
from pathlib import Path
from typing import Callable, Dict, Optional, Union
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow

from .data import InputFunction
from .data.graph2mol import MoleculeData
from .estimator import MolGANTrainHooksFn
from .molgan import MolGANParameters, MolGANEstimator

MakeModelFn = Callable[[Namespace, MoleculeData, MolGANTrainHooksFn], MolGANEstimator]
MakeTrainFn = Callable[[DataFlow, int, int, MoleculeData], InputFunction]
PredictFn = Callable[[], Union[tf.Tensor, Dict[str, tf.Tensor]]]
MakePredictFn = Callable[[Path, int, int, Optional[int]], PredictFn]


class Experiment:

    def __init__(self,
                 make_model_fn: MakeModelFn,
                 make_train_fn: MakeTrainFn,
                 make_predict_fn: MakePredictFn) -> None:
        self._model_fn = make_model_fn
        self._train_fn = make_train_fn
        self._predict_fn = make_predict_fn

    @property
    def make_model_fn(self):
        return self._model_fn

    @property
    def make_train_fn(self):
        return self._train_fn

    @property
    def make_predict_fn(self):
        return self._predict_fn


def get_params_from_args(args, data_params):
    params = MolGANParameters(
        max_nodes=data_params.MAX_NODES,
        num_node_types=data_params.NUM_NODE_TYPES,
        num_edge_types=data_params.NUM_EDGE_TYPES,
        num_latent=args.num_latent,
        with_value_net=args.weight_reward_loss > 0,
        lam=1.0,
        generator_learning_rate=args.generator_learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        gradient_penalty_weight=args.weight_gradient_penalty,
        connectivity_penalty_weight=args.connectivity_penalty_weight,
        valence_penalty_weight=args.valence_penalty_weight,
        variance_penalty_weight=args.variance_penalty_weight,
        batch_size=args.batch_size,
        discriminator_gcn_units=(128, 128),
        discriminator_dense_units=(128, 64),
        generator_units=(128, 256, 512),
        without_cycle_discriminator=args.without_cycle_discriminator,
        without_unary_discriminator=args.without_unary_discriminator,
        without_gcn_skip_connections=args.without_gcn_skip_connections,
        without_generator_skip_connections=args.without_generator_skip_connections,
        without_gated_gcn=args.without_gated_gcn,
        softmax_temp=args.temperature,
        add_summaries=True,
        summarize_gradients=args.summarize_gradients,
    )
    return params


class WGanExperiment(Experiment):

    def __init__(self):
        super(WGanExperiment, self).__init__(
            make_model_fn=self._make_gan_model,
            make_predict_fn=self._make_predict_fn,
            make_train_fn=self._make_train_fn,
        )

    def _make_gan_model(self, args, data_params, hooks_fn):
        from gan.mol.molgan import MolGANEstimator

        params = get_params_from_args(args, data_params)
        model = MolGANEstimator(hooks_fn=hooks_fn, params=params)
        return model

    def _make_predict_fn(self, data_dir, num_latent, n_samples, batch_size, seed=7245):
        from gan.mol.data import make_predict_fn

        return make_predict_fn(data_dir, num_latent, n_samples, batch_size, seed=seed)

    def _make_train_fn(self, ds, batch_size, num_latent, data_params):
        from gan.mol.data import InputFunction

        return InputFunction(ds, batch_size, num_latent, data_params)


class NonSaturatingGanExperiment(Experiment):

    def __init__(self):
        super(NonSaturatingGanExperiment, self).__init__(
            make_model_fn=self._make_gan_model,
            make_predict_fn=self._make_predict_fn,
            make_train_fn=self._make_train_fn,
        )

    def _make_gan_model(self, args, data_params, hooks_fn):
        from gan.mol.non_saturating_molgan import NonSaturatingMolGANEstimator

        params = get_params_from_args(args, data_params)
        model = NonSaturatingMolGANEstimator(hooks_fn=hooks_fn, params=params)
        return model

    def _make_predict_fn(self, data_dir, num_latent, n_samples, batch_size, seed=7245):
        from gan.mol.data import make_predict_fn

        return make_predict_fn(data_dir, num_latent, n_samples, batch_size, seed=seed)

    def _make_train_fn(self, ds, batch_size, num_latent, data_params):
        from gan.mol.data import InputFunction

        return InputFunction(ds, batch_size, num_latent, data_params)


class AliceExperiment(Experiment):

    def __init__(self):
        super(AliceExperiment, self).__init__(
            make_model_fn=self._make_gan_model,
            make_predict_fn=self._make_predict_fn,
            make_train_fn=self._make_train_fn,
        )
        self._data_params = None

    def _make_gan_model(self, args, data_params, hooks_fn):
        from gan.mol.alice import MolGANALICEEstimator

        self._data_params = data_params
        params = get_params_from_args(args, data_params)
        model = MolGANALICEEstimator(hooks_fn=hooks_fn, params=params)
        return model

    def _make_predict_fn(self, data_dir, num_latent, n_samples, batch_size, seed=7245):
        from gan.mol.alice.data import PredictionInputFunction

        factory = PredictionInputFunction(data_dir, num_latent, self._data_params)
        return factory.create(n_samples, batch_size, seed)

    def _make_train_fn(self, ds, batch_size, num_latent, data_params):
        from gan.mol.alice.data import InputFunction

        return InputFunction(ds, batch_size, num_latent, data_params)
