"""
前処理後のデータを用いて学習を行うモジュール.
"""
import os
import argparse
from config import common_args, Parameters
from utils import dump_params, setup_params
from utils import set_logging
import logging

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AdamW

def train(params):
    model = AutoModelForSequenceClassification.from_pretrained(params.model_name, num_labels = 4)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    sequences = [
        "develop cut edg applic that perform",
        "highli collabor environ with cross",
        "strong abil work requir includ"
    ]
    encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
    print(encoded_sequences)



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
