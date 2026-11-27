set -euo pipefail
cd -- "$(dirname -- "$0")"
modelscope download --model google/vit-base-patch16-224  --local_dir ./vit-base-patch16-224
