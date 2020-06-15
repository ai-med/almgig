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
from collections import OrderedDict
import json
import logging
import guacamol
from guacamol.assess_distribution_learning import _evaluate_distribution_learning_benchmarks
from guacamol.distribution_learning_benchmark import ValidityBenchmark, \
    UniquenessBenchmark
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.standard_benchmarks import novelty_benchmark, kldiv_benchmark
from guacamol.utils.data import get_time_string

LOG = logging.getLogger(__name__)


# see guacamol.assess_distribution_learning._assess_distribution_learning
def assess_distribution_learning(model: DistributionMatchingGenerator,
                                 training_file_path: str,
                                 json_output_file: str,
                                 number_samples: int) -> None:
    LOG.info('Benchmarking distribution learning')
    benchmarks = [
        ValidityBenchmark(number_samples=number_samples),
        UniquenessBenchmark(number_samples=number_samples),
        novelty_benchmark(training_set_file=training_file_path, number_samples=number_samples),
        kldiv_benchmark(training_set_file=training_file_path, number_samples=number_samples),
    ]

    results = _evaluate_distribution_learning_benchmarks(model=model, benchmarks=benchmarks)

    benchmark_results = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    LOG.info('Save results to file %s', json_output_file)
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))
