# FedSCAN

English | [简体中文](README_zh-CN.md)

**Spectral-Constrained Robust Aggregation for Secure LoRA-Based Federated Learning**
KDD 2027 · [Paper](https://openreview.net/forum?id=YCna6pqYJk)

FedSCAN is a two-stage server-side aggregator for LoRA-based federated learning. Stage I estimates a robust update through iterative client reweighting. Stage II clips blockwise singular values before applying the aggregated update. The framework includes CLIP-LoRA for image-text retrieval and ViT-LoRA for image classification.

![FedSCAN overview](assets/overview.png)

[Vector overview](assets/overview.pdf)

## Contents

- `fedscan/core/`: FedSCAN aggregation and clean-client federated training.
- `fedscan/models/`: CLIP, ViT, and LoRA update utilities.
- `fedscan/data/`: federated image-text and classification data loaders.
- `fedscan/config/`: model, training, and aggregation settings.
- `fedscan/utils/`: retrieval and classification evaluation.
- `datasets/`: dataset directories and download scripts.

The release excludes attack implementations, experiment launchers, datasets, checkpoints, and experimental logs.

## Installation

Python 3.10 or later is required.

`pip install -r requirements.txt`

`pip install -e .`

## Interface

The core entry point is `fedscan.FedSCANAggregator`. Its `aggregate` method accepts a list of parameter-name-to-tensor dictionaries and returns the aggregated update and metrics. Stage modes 1, 2, and 3 select the full method, Stage I only, and averaging followed by spectral clipping.

Defaults are configured in `fedscan/config/fedscan_config.py`. The supplied implementation initializes stored spectral statistics through `aggregate(..., is_init_benign=True, clean_updates=...)`. The clean-client server performs this initialization in its configured first round.

`fedscan.core.federated_server.FederatedServer` provides the training interface. Set dataset paths and configuration values before constructing it; no training is launched on import.

## Data

Download scripts are retained under `datasets/flickr30k/`, `datasets/coco/`, `datasets/tiny-imagenet/`, and `datasets/cifar100/`. Invoke them with `bash`; no data are bundled. The existing COCO script downloads annotations only; place the images separately. Model download helpers are under `datasets/google/` and `datasets/openai/`. The CIFAR-100 and ViT helpers use ModelScope; install the optional download dependencies with `pip install -e '.[downloads]'`.

Training JSON maps `client_1`, `client_2`, and subsequent client identifiers to sample lists. Validation JSON uses `val_data`. Retrieval samples contain `img_path` and `text`; classification samples contain `img_path` and `label`. Store prepared JSON files in `datasets/data_cache/` and configure pretrained-model locations in `fedscan/config/base_config.py`.

## Citation

```bibtex
@inproceedings{hu2027fedscan,
  title={Spectral-Constrained Robust Aggregation for Secure LoRA-Based Federated Learning},
  author={Hu, Wenhan and Wang, Zhe and Huang, Rui and Yan, Gang},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2027}
}
```

## License

Apache-2.0. Original attribution and release changes are retained in `NOTICE`.
