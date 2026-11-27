#!/usr/bin/env bash
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail
cd -- "$(dirname -- "$0")"
set -e

python -m pip install -U modelscope

modelscope download \
  --dataset OpenDataLab/CIFAR-100 \
  --local_dir .
