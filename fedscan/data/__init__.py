from .clip_dataload_utils import FederatedDataLoader, FederatedDataset
from .vit_dataload_utils import FederatedClsDataLoader, FederatedClsDataset
import json

def _task_from_model(model_name: str) -> str:
    u = (model_name or '').upper()
    if 'CLIP' in u:
        return 'clip'
    if 'VIT' in u:
        return 'cls'
    raise ValueError(f"Unknown model_name '{model_name}', cannot infer task (expect CLIP*/ViT*).")

def get_dataset(federated_data_path: str, model_name: str=None, processor=None, client_id=None):
    assert model_name is not None, 'model_name must be provided to route dataset.'
    with open(federated_data_path, 'r') as f:
        jf = json.load(f)
    task = _task_from_model(model_name)
    if client_id is not None and processor is None:
        if task == 'cls':
            return FederatedClsDataLoader(client_id=client_id, client_data_path=federated_data_path)
        elif task == 'clip':
            return FederatedDataLoader(client_id=client_id, client_data_path=federated_data_path)
    if processor is not None and client_id is None:
        val_list = jf.get('val_data', [])
        if task == 'cls':
            return FederatedClsDataset(val_list, processor=processor)
        elif task == 'clip':
            return FederatedDataset(val_list, processor=processor)
    raise NotImplementedError('Invalid get_dataset call: need (client_id XOR processor). And model_name must be provided to route dataset.')
