#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from fedscan.data.vit_dataload_utils import FederatedClsDataset
from fedscan.config.base_config import vit_lora as lora_cfg
from tqdm import tqdm

@torch.no_grad()
def evaluate_cls(model, eval_data_path: str, device, topk=(1,)):
    model.to(device)
    model.eval()
    vit_name = lora_cfg.get('vit_pre_model_name', 'google/vit-base-patch16-224')
    processor = AutoImageProcessor.from_pretrained(vit_name)
    with open(eval_data_path, 'r') as f:
        jf = json.load(f)
    val_list = jf['val_data']
    ds = FederatedClsDataset(val_list, processor=processor)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    correct_at_k = {k: 0 for k in topk}
    total = 0
    for batch in tqdm(loader, desc='Evaluating (CLS)'):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(pixel_values=pixel_values, return_loss=False)
        logits = outputs['logits']
        total += labels.size(0)
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        for k in topk:
            correct_at_k[k] += correct[:k].reshape(-1).float().sum().item()
    accs = {f'top{k}': correct_at_k[k] / total if total > 0 else 0.0 for k in topk}
    return accs
