"""
前処理後のデータを用いて学習を行うモジュール.
"""
import os
import argparse
from config import common_args, Parameters
from utils import dump_params, setup_params
from utils import set_logging
import logging

import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AdamW

def train(params):
    model = Classifier(params.model_name, num_classes=params.classes_num)
    model = model.to(params.device)
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    sequences = [
        "develop cut edg applic that perform",
        "highli collabor environ with cross",
        "strong abil work requir includ"
    ]
    encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')


class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output, _ = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict=False) # Pythonの実行上必要なので加筆しました。
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)  # コマンドライン引数引数を読み込み
    # parser.add_argument("--main")  # 実行スクリプト固有のコマンドライン引数があればここに記入する．
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    train(params)
