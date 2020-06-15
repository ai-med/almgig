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
import tensorflow as tf


def setup_logging(level):
    logger = logging.getLogger('nevae')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level)

    tf.logging.set_verbosity(level)
