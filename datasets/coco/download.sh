set -euo pipefail
cd -- "$(dirname -- "$0")"
wget -c  \
  -O annotations_trainval2014.zip http://images.cocodataset.org/annotations/annotations_trainval2014.zip


unzip annotations_trainval2014.zip
