#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from tqdm import tqdm
from fedscan.models.lora_utils import extract_lora_update
from fedscan.config.base_config import train as train_cfg, federated as fl_cfg
from fedscan.models import get_model
from fedscan.data import get_dataset

class FederatedClient:

    def __init__(self, client_id, model_name, global_lora_path, federated_data_path, local_epochs):
        self.client_id = client_id
        self.device = torch.device(train_cfg['device'])
        self.global_model = get_model(model_name, global_lora_path=global_lora_path)
        self.local_model = get_model(model_name, global_lora_path=global_lora_path)
        self.local_model.to(self.device)
        self.local_model.train()
        self.local_epochs = local_epochs
        self.task = 'clip' if 'CLIP' in model_name.upper() else 'vit'
        self.optimizer = torch.optim.AdamW(self.local_model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
        self.data_loader = get_dataset(federated_data_path=federated_data_path, model_name=model_name, client_id=client_id).get_loader(batch_size=fl_cfg['batch_size'])

    def local_train(self):
        total_loss = 0.0
        accumulation_steps = fl_cfg['accumulation_steps']
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc=f'Client {self.client_id} Epoch {epoch + 1}')):
                if self.task == 'clip':
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.local_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=True)
                else:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.local_model(pixel_values=pixel_values, labels=labels, return_loss=True)
                loss = outputs['loss']
                epoch_loss += float(loss.item())
                (loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            if len(self.data_loader) % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            avg_epoch_loss = epoch_loss / max(len(self.data_loader), 1)
            total_loss += avg_epoch_loss
        local_lora_path = f'./tmp/client_{self.client_id}_lora'
        avg_total_loss = total_loss / max(self.local_epochs, 1)
        return (local_lora_path, avg_total_loss)

    def get_update(self):
        return extract_lora_update(self.global_model, self.local_model)
