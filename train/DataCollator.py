from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List
import torch
import re

class DataCollatorForLabelOnly:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Tokenize as usual
        batch = self.tokenizer([ex["Text"] for ex in examples], padding=True, truncation=True, return_tensors="pt")
        labels = batch["input_ids"].clone()

        # For each example, mask out everything except the label token(s)
        for i, ex in enumerate(examples):
            text = ex["Text"]
            # Find the label (assume it's the last number before <|end_of_text|>)
            match = re.search(r"\n([^\n<]+)<\|end_of_text\|>", text)
            if not match:
                raise ValueError(f"Could not find label in: {text}")
            label_str = match.group(1).strip()
            # Tokenize the label string alone
            label_tokens = self.tokenizer(label_str, add_special_tokens=False)["input_ids"]
            eos_token = self.tokenizer("<|end_of_text|>", add_special_tokens=False)["input_ids"]

            # Tokenize the full text
            full_tokens = batch["input_ids"][i].tolist()

            # Find the position where label_tokens appear just before eos_token
            for j in range(len(full_tokens) - len(label_tokens) - len(eos_token) + 1):
                if (full_tokens[j:j+len(label_tokens)] == label_tokens and
                    full_tokens[j+len(label_tokens):j+len(label_tokens)+len(eos_token)] == eos_token):
                    label_start_pos = j
                    break
            else:
                raise ValueError(f"Could not find label tokens in: {text}")

            # Set all positions except label token(s) to -100
            labels[i, :] = -100
            labels[i, label_start_pos:label_start_pos+len(label_tokens)] = batch["input_ids"][i, label_start_pos:label_start_pos+len(label_tokens)]

        batch["labels"] = labels
        return batch