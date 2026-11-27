# FedSCAN

[English](README.md) | 简体中文

**Spectral-Constrained Robust Aggregation for Secure LoRA-Based Federated Learning**  
**面向安全 LoRA 联邦学习的谱约束鲁棒聚合**  
KDD 2027 · [论文](https://openreview.net/forum?id=YCna6pqYJk)

FedSCAN 是面向 LoRA 联邦学习的两阶段服务器端聚合方法。Stage I 通过迭代客户端重加权估计鲁棒更新；Stage II 对各矩阵块的奇异值进行裁剪，再应用聚合更新。项目包含用于图文检索的 CLIP-LoRA 和用于图像分类的 ViT-LoRA。

![FedSCAN 方法总览](assets/overview.png)

[矢量版总览图](assets/overview.pdf)

## 目录

- `fedscan/core/`：FedSCAN 聚合及良性客户端联邦训练。
- `fedscan/models/`：CLIP、ViT 和 LoRA 更新工具。
- `fedscan/data/`：联邦图文与分类数据加载器。
- `fedscan/config/`：模型、训练和聚合配置。
- `fedscan/utils/`：检索与分类评估。
- `datasets/`：数据目录和下载脚本。

本仓库不包含攻击实现、实验启动脚本、数据集、模型权重、检查点或实验日志。

## 安装

需要 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
pip install -e .
```

## 接口

核心入口为 `fedscan.FedSCANAggregator`。`aggregate` 接收由“参数名到张量”的字典组成的客户端更新列表，返回聚合更新和指标。`stage_mode` 取 1、2、3 时，分别使用完整方法、仅 Stage I、平均聚合后进行谱裁剪。

默认参数位于 `fedscan/config/fedscan_config.py`。当前实现通过 `aggregate(..., is_init_benign=True, clean_updates=...)` 初始化并保存谱统计量；良性客户端服务器在配置的首轮完成该初始化。

`fedscan.core.federated_server.FederatedServer` 提供训练接口。创建服务器前需设置数据路径和配置；导入模块不会自动启动训练。

## 数据

`datasets/flickr30k/`、`datasets/coco/`、`datasets/tiny-imagenet/` 和 `datasets/cifar100/` 保留下载脚本，使用 `bash` 调用，仓库不附带数据。现有 COCO 脚本只下载标注，图像需另行准备。模型下载脚本位于 `datasets/google/` 和 `datasets/openai/`。CIFAR-100 和 ViT 下载脚本使用 ModelScope，可通过以下命令安装可选下载依赖：

```bash
pip install -e '.[downloads]'
```

训练 JSON 将 `client_1`、`client_2` 等客户端标识映射到样本列表，验证 JSON 使用 `val_data`。检索样本包含 `img_path` 和 `text`，分类样本包含 `img_path` 和 `label`。准备好的 JSON 放入 `datasets/data_cache/`，预训练模型路径在 `fedscan/config/base_config.py` 中配置。

## 引用

```bibtex
@inproceedings{hu2027fedscan,
  title={Spectral-Constrained Robust Aggregation for Secure LoRA-Based Federated Learning},
  author={Hu, Wenhan and Wang, Zhe and Huang, Rui and Yan, Gang},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2027}
}
```

## 许可

采用 Apache-2.0 许可证，原始署名和发布变更说明保留在 `NOTICE` 中。
