import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import AutoTokenizer

@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None

class SST2Tokenizer:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return enc

def sst2_collate(batch: List[Dict[str, Any]], tokenizer: SST2Tokenizer) -> TextBatch:
    texts = [ex["text"] for ex in batch]
    labels = [ex.get("label", -100) for ex in batch]
    enc = tokenizer(texts)
    labels = torch.tensor(labels, dtype=torch.long)
    return TextBatch(enc["input_ids"], enc["attention_mask"], labels)
