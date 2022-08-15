
"""
model作成
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F


# BERT-model
class Classifier_Conv(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.cnn1 = nn.Conv1d(self.config.hidden_size, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, 4, kernel_size=2, padding=1)
        # self.linear = nn.Linear(768, num_classes)
        # nn.init.normal_(self.linear.weight, std=0.02)
        # nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict=True) # Pythonの実行上必要なので加筆しました。
        last_hidden_state = output['last_hidden_state'].permute(0, 2, 1)
        output = self.dropout(output)
        output = F.relu(self.cnn1(last_hidden_state))
        output = self.dropout(output)
        output = self.cnn2(output)
        output, _ = torch.max(output, 2)
        return output