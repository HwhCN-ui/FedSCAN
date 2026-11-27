#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from fedscan.config.base_config import lora as lora_cfg, vit_lora as vit_lora_cfg

def _is_target_lora_key(k: str) -> bool:
    if not (k.endswith('lora_A.default.weight') or k.endswith('lora_B.default.weight')):
        return False
    for tm in lora_cfg['target_modules']:
        if f'.{tm}.' in k or k.endswith(f'.{tm}.lora_A.default.weight') or k.endswith(f'.{tm}.lora_B.default.weight'):
            return True
    for tm in vit_lora_cfg['target_modules']:
        if f'.{tm}.' in k or k.endswith(f'.{tm}.lora_A.default.weight') or k.endswith(f'.{tm}.lora_B.default.weight'):
            return True
    return False

def extract_lora_update(global_model, local_model):
    update = {}
    g = global_model.lora_model.state_dict()
    l = local_model.lora_model.state_dict()
    for k, lv in l.items():
        if not _is_target_lora_key(k):
            continue
        gv = g.get(k, None)
        if gv is None:
            update[k] = lv.clone()
        else:
            lv_t = lv.detach().cpu()
            gv_t = gv.detach().cpu()
            update[k] = lv_t - gv_t
    return update

def apply_lora_update(global_model, update):
    g = global_model.lora_model.state_dict()
    for k, dv in update.items():
        if k in g:
            g[k] = g[k] + dv.to(g[k].dtype).to(g[k].device)
        else:
            pass
    global_model.lora_model.load_state_dict(g, strict=False)
    return global_model

def compute_update_norm(client_update, intermediate_update):
    keys = set(client_update.keys()) | set(intermediate_update.keys())
    total = 0.0
    for k in keys:
        a = client_update.get(k, None)
        b = intermediate_update.get(k, None)
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        diff = a - b
        total += torch.sum(diff * diff).item()
    return total

def compute_lora_norm(model):
    total_norm = 0.0
    sd = model.lora_model.state_dict()
    for k, v in sd.items():
        if _is_target_lora_key(k):
            total_norm += torch.norm(v, p='fro').item()
    return total_norm
