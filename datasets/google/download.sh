#!/usr/bin/env bash
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail
cd -- "$(dirname -- "$0")"
modelscope download --model google/vit-base-patch16-224  --local_dir ./vit-base-patch16-224
