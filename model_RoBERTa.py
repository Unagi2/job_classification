
"""
model作成
"""
import torch.nn as nn
from transformers import AdamW, AutoModel, AutoTokenizer


# BERT-model
class Classifier_RoBERTa(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask):
        output, _ = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict=False) 
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear(output)
        return output