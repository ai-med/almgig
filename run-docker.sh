#!/bin/bash

docker run -it --rm --gpus all \
  -v $PWD:/workbench \
  almgig \
  $@
