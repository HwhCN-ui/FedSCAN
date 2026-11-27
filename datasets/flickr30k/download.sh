set -euo pipefail
cd -- "$(dirname -- "$0")"
echo "开始下载 Flickr30k 数据集..."

wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"

echo "合并分卷..."
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip

echo "清理分卷文件..."
rm flickr30k_part00 flickr30k_part01 flickr30k_part02

echo "解压数据集..."
unzip -q flickr30k.zip -d ./flickr30k

echo "删除压缩包..."
rm flickr30k.zip

echo "✅ 下载并解压 Flickr30k 数据集成功！"
