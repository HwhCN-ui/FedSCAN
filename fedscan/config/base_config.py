#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from datetime import datetime
import os
federated = {'num_clients': 20, 'num_rounds': 30, 'clients_per_round': 20, 'local_epochs': 1, 'batch_size': 128, 'num_workers': 4, 'prefetch_factor': 8, 'accumulation_steps': 1}
date_str = datetime.now().strftime('%Y%m%d_%H%M')
lora = {'pre_model_name': './datasets/openai/clip-vit-base-patch32', 'rank': 16, 'alpha': 32, 'target_modules': ['q_proj', 'k_proj', 'v_proj'], 'dropout': 0.05, 'global_lora_path': f'./models/{date_str}/global_lora/initial'}
vit_lora = {'vit_pre_model_name': './datasets/google/vit-base-patch16-224', 'rank': 16, 'alpha': 32, 'target_modules': ['query', 'key', 'value', 'classifier'], 'dropout': 0.05, 'global_lora_path': f'./models/{date_str}/global_lora/initial', 'num_labels': 200}
train = {'lr': 0.001, 'weight_decay': 1e-05, 'device': 'cuda:0'}
log = {'log_dir': os.path.join('./logs', date_str), 'log_name': 'fedscan_fl.log'}
tensorboard = {'log_dir': os.path.join('./run/fedscan_fl_exp', date_str), 'log_freq': 1}
