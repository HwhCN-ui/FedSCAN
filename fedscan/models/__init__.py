#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

def get_model(model_name, global_lora_path=None):
    if model_name == "clip":
        from .clip_lora import CLIPLoRA
        model_class = CLIPLoRA
    elif model_name == "vit":
        from .vit_lora import ViTLoRAClassifier
        model_class = ViTLoRAClassifier
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_class() if global_lora_path is None else model_class.from_pretrained(global_lora_path)
