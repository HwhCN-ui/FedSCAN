import torch
import torch.nn.functional as F
from tqdm import tqdm
from fedscan.data.clip_dataload_utils import FederatedDataset
from transformers import CLIPProcessor
from fedscan.config.base_config import lora as lora_cfg
from fedscan.data import get_dataset

def evaluate(model, eval_data_path, device, model_name):
    model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(lora_cfg['pre_model_name'])
    total_correct = 0
    total_samples = 0
    dataset = get_dataset(eval_data_path, model_name=model_name, processor=processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
            logits = outputs['logits_per_image']
            preds = torch.argmax(logits, dim=1)
            gt = torch.arange(preds.size(0), device=preds.device)
            total_correct += torch.sum(preds == gt).item()
            total_samples += preds.size(0)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return acc
