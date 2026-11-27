#!/usr/bin/env bash
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail
cd -- "$(dirname -- "$0")"
python -c 'from huggingface_hub import snapshot_download; snapshot_download("openai/clip-vit-base-patch32", local_dir="clip-vit-base-patch32")'
