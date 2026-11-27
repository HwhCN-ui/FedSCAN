set -euo pipefail
cd -- "$(dirname -- "$0")"
set -e

python -m pip install -U modelscope

modelscope download \
  --dataset OpenDataLab/CIFAR-100 \
  --local_dir .
