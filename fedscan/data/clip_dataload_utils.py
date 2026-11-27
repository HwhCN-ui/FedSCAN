import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from fedscan.config.base_config import lora as lora_cfg, federated as fed_cfg

class FederatedDataset(Dataset):

    def __init__(self, client_data, processor):
        self.data = client_data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        from PIL import Image
        img = Image.open(item['img_path']).convert('RGB')
        enc = self.processor(images=img, text=item['text'], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        pixel_values = enc['pixel_values'].squeeze(0)
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'text': item['text']}

class FederatedDataLoader:

    def __init__(self, client_id, client_data_path):
        self.client_id = client_id
        self.data_path = client_data_path
        self.processor = CLIPProcessor.from_pretrained(lora_cfg['pre_model_name'])

    def get_loader(self, batch_size=16, shuffle=True):
        import json
        with open(self.data_path, 'r') as f:
            client_data = json.load(f)[f'client_{self.client_id}']
        dataset = FederatedDataset(client_data, self.processor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=fed_cfg['num_workers'], prefetch_factor=fed_cfg['prefetch_factor'])
