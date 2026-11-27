#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from peft import LoraConfig, get_peft_model, PeftModel
from peft import TaskType
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
from fedscan.config.base_config import lora as lora_cfg

class CLIPLoRA(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = CLIPModel.from_pretrained(lora_cfg['pre_model_name'])
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.lora_cfg = LoraConfig(r=lora_cfg['rank'], lora_alpha=lora_cfg['alpha'], target_modules=lora_cfg['target_modules'], lora_dropout=lora_cfg['dropout'], bias='none', task_type=TaskType.FEATURE_EXTRACTION)
        self.lora_model = get_peft_model(self.base_model, self.lora_cfg)
        self.processor = CLIPProcessor.from_pretrained(lora_cfg['pre_model_name'])
        self.lora_model.print_trainable_parameters()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, return_loss=False):
        outputs = self.lora_model.base_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=return_loss, return_dict=True)
        return outputs

    def save_lora(self, save_path):
        self.lora_model.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, lora_path):
        model = cls()
        model.lora_model = PeftModel.from_pretrained(model.base_model, lora_path, is_trainable=True)
        return model
