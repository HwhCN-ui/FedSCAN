#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import os, json, random
from typing import Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from fedscan.config.base_config import vit_lora as lora_cfg, federated as fed_cfg

class FederatedClsDataset(Dataset):

    def __init__(self, data_list: List[Dict], processor=None):
        self.data = data_list
        vit_name = lora_cfg.get('vit_pre_model_name', 'google/vit-base-patch16-224')
        self.processor = processor or AutoImageProcessor.from_pretrained(vit_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img = Image.open(rec['img_path']).convert('RGB')
        enc = self.processor(images=img, return_tensors='pt')
        pixel_values = enc['pixel_values'].squeeze(0)
        label = int(rec['label'])
        return {'pixel_values': pixel_values, 'labels': torch.tensor(label, dtype=torch.long)}

class FederatedClsDataLoader:

    def __init__(self, client_id: int, client_data_path: str):
        self.client_id = client_id
        self.data_path = client_data_path
        vit_name = lora_cfg.get('vit_pre_model_name', 'google/vit-base-patch16-224')
        self.processor = AutoImageProcessor.from_pretrained(vit_name)

    def get_loader(self, batch_size=16, shuffle=True, num_workers=2):
        with open(self.data_path, 'r') as f:
            jf = json.load(f)
        data = jf[f'client_{self.client_id}']
        ds = FederatedClsDataset(data, processor=self.processor)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=fed_cfg['num_workers'], prefetch_factor=fed_cfg['prefetch_factor'])
