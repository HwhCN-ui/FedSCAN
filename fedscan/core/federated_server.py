import logging
from pathlib import Path
import numpy as np
import torch
from fedscan.config.base_config import federated as fl_cfg, lora as lora_cfg, train as train_cfg, vit_lora as vit_cfg
from fedscan.config.fedscan_config import stage2 as stage2_cfg
from fedscan.core.federated_client import FederatedClient
from fedscan.core.fedscan_aggregator import FedSCANAggregator
from fedscan.models import get_model
from fedscan.models.lora_utils import apply_lora_update
from fedscan.utils.evaluator import evaluate
from fedscan.utils.evaluator_vit import evaluate_cls

class FederatedServer:

    def __init__(self, federated_train_data_path, federated_val_data_path, model_name='clip', fedscan_stage=1, fedscan_c=1.0, fedscan_lambda=None, fedscan_delta=None, clients_per_round=None, diagnostics=False):
        if model_name not in ('clip', 'vit'):
            raise ValueError("model_name must be 'clip' or 'vit'")
        self.model_name = model_name
        self.device = torch.device(train_cfg['device'])
        self.num_clients = fl_cfg['num_clients']
        self.num_rounds = fl_cfg['num_rounds']
        self.local_epochs = fl_cfg['local_epochs']
        self.clients_per_round = fl_cfg['clients_per_round'] if clients_per_round is None else int(clients_per_round)
        if not 1 <= self.clients_per_round <= self.num_clients:
            raise ValueError('clients_per_round must be between 1 and num_clients')
        self.federated_train_data_path = federated_train_data_path
        self.federated_val_data_path = federated_val_data_path
        self.global_model = get_model(model_name)
        config = lora_cfg if model_name == 'clip' else vit_cfg
        self.global_lora_path = config['global_lora_path']
        Path(self.global_lora_path).parent.mkdir(parents=True, exist_ok=True)
        self.global_model.save_lora(self.global_lora_path)
        self.aggregator = FedSCANAggregator(stage_mode=fedscan_stage, fedscan_c=fedscan_c, fedscan_lambda=fedscan_lambda, fedscan_delta=fedscan_delta, diagnostics=diagnostics)
        self.logger = logging.getLogger('FedSCAN')
        self.current_round = 0
        self.history = []

    def select_clients(self):
        return np.random.choice(np.arange(1, self.num_clients + 1), self.clients_per_round, replace=False).tolist()

    def collect_updates(self, client_ids):
        updates, losses = ([], [])
        for client_id in client_ids:
            client = FederatedClient(client_id=client_id, model_name=self.model_name, global_lora_path=self.global_lora_path, federated_data_path=self.federated_train_data_path, local_epochs=self.local_epochs)
            _, loss = client.local_train()
            updates.append(client.get_update())
            losses.append(loss)
        return (updates, float(np.mean(losses)))

    def update_global_model(self, update):
        self.global_model = get_model(self.model_name, global_lora_path=self.global_lora_path)
        self.global_model = apply_lora_update(self.global_model, update)
        path = Path(self.global_lora_path).parent / f'round_{self.current_round}'
        self.global_model.save_lora(str(path))
        self.global_lora_path = str(path)

    def evaluate_model(self):
        if self.model_name == 'vit':
            return evaluate_cls(self.global_model, self.federated_val_data_path, self.device, topk=(1,))['top1']
        return evaluate(self.global_model, self.federated_val_data_path, self.device, self.model_name)

    def run(self):
        for round_index in range(self.num_rounds):
            self.current_round = round_index + 1
            client_ids = self.select_clients()
            updates, loss = self.collect_updates(client_ids)
            initialize = round_index == stage2_cfg['benign_init_round']
            update, metrics = self.aggregator.aggregate(updates, is_init_benign=initialize, clean_updates=updates if initialize else None)
            self.update_global_model(update)
            row = {'round': self.current_round, 'client_ids': client_ids, 'loss': loss, 'accuracy': self.evaluate_model(), **metrics}
            self.history.append(row)
            self.logger.info('Round %s: loss=%.6f accuracy=%.6f', self.current_round, loss, row['accuracy'])
        return self.history
