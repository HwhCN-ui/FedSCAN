#!/usr/bin/env bash
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail
cd -- "$(dirname -- "$0")"
set -euo pipefail

URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
OUT="tiny-imagenet-200.zip"

wget -c --show-progress -O "$OUT" "$URL"

unzip -o "$OUT"

test -d tiny-imagenet-200/train && echo "✓ 解压完成：tiny-imagenet-200/" || echo "× 解压目录不存在"
