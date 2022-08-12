import os
import re
import shutil
from os.path import isdir
import argparse
import numpy as np
from config import common_args, Parameters
from utils import setup_params, get_device
import logging
import pandas as pd
from typing import Any
import torch
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def gen_model(params, result_dir) -> None:
    """データセット作成と前処理

    データセットのロードとリサンプリングを処理

    Args:
        params: パラメータ
        result_dir: ディレクトリ

    Returns:
        None

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    logger.info('Loading Test Dataset...')

    torch.manual_seed(params.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium', pad_token_id=tokenizer.eos_token_id).cuda()
    model.resize_token_embeddings(len(tokenizer))

    # 読み込み
    descriptions = pd.read_csv(params.train_file_path)['description']
    # 不要タグの削除-クリーニング
    descriptions = descriptions.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())
    descriptions = descriptions.sample(100, random_state=params.seed)  # サンプル数削減(frac = 0.1)
    max_length = max([len(tokenizer.encode(description)) for description in descriptions])

    dataset = TrainDataset(descriptions, tokenizer, max_length=max_length)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    import gc
    gc.collect()

    logger.info('Training...')
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 学習パラメータ
    training_args = TrainingArguments(output_dir=result_dir+"/", num_train_epochs=5, logging_steps=5000, save_steps=5000,
                                      per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                      warmup_steps=100, weight_decay=0.01, logging_dir=result_dir+"/", report_to='none')

    # 学習
    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                        'attention_mask': torch.stack([f[1] for f in data]),
                                        'labels': torch.stack([f[0] for f in data])}).train()
    # .save_model(result_dir + f'/out_model')
    torch.cuda.empty_cache()

    # 生成モデルからテキスト生成
    generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids.cuda()

    sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                    max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    print("============================================")

    #######################################################################################
    # 元データセットへの生成データの追加
    df = pd.read_csv(params.train_file_path)  # 読み込み

    label_num = params.num_classes  # ラベル番号
    index_num = df['id'].tail()

    # 不要タグの削除-クリーニング
    df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())

    df_save = df
    print(df_save.head())

    # 先頭３単語抽出
    df['description'] = df['description'].apply(lambda x: " ".join(re.findall(r"[a-zA-Z]+", x)[0:3]))

    for i in range(1, label_num):
        print("============================================")
        print("label: " + str(i))
        print("============================================")
        for n, text in enumerate(df[df['jobflag'] == i].description):
            print("label: " + str(i) + "  description-num: " + str(n))
            print("============================================")
            df_size = df[df['jobflag'] == i].description.size  # 1job当たりのデータサイズ
            gen_num = int((params.sampling_num - df_size) / df_size)  # 生成回数
            txt_len_min = min(df[df['jobflag'] == i].description.str.len())
            txt_len_max = max(df[df['jobflag'] == i].description.str.len())
            df_txt_len = df[df['jobflag'] == i].description.str.len().median()  # 生成文字列数

            # ターゲットテキストの指定
            # target_text = text.split('.', 2)[0]  # + "."
            target_text = text

            # 生成準備
            generated = tokenizer(target_text, return_tensors="pt").input_ids.cuda()

            # テキスト生成設定
            sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                            num_beams=gen_num, no_repeat_ngram_size=2, early_stopping=True,
                                            min_length=int(txt_len_min), max_length=int(txt_len_max), top_p=0.95,
                                            num_return_sequences=gen_num)

            # 生成データ
            for s, sample_output in enumerate(sample_outputs):
                # データフレームに追加 strip("'").
                list_add = [[s + 1, tokenizer.decode(sample_output, skip_special_tokens=True).strip().replace('\n', " "), i]]
                # print(list_add)

                df_add = pd.DataFrame(data=list_add, columns=['id', 'description', 'jobflag'])
                # print(df_add)

                print("{}: {}".format(s, tokenizer.decode(sample_output, skip_special_tokens=True).strip()))

                # Concat処理　元データセットの末尾に追加
                logger.info("Saving train_generated.csv...")
                df_over = pd.concat([df_save, df_add], axis=0, ignore_index=True).reset_index()

            print("============================================")
            # logger.info(text)

    df_over.to_csv('dataset/train_generated.csv')


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)  # コマンドライン引数引数を読み込み
    # parser.add_argument("--main")  # 実行スクリプト固有のコマンドライン引数があればここに記入する．
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    vars(params).update({'device': str(get_device())})  # 空きGPU検索

    # 結果出力用ファイルの作成
    result_dir = f'./result/{params.run_date}'  # 結果出力ディレクトリ

    os.mkdir(result_dir)  # 実行日時を名前とするディレクトリを作成
    # os.mkdir(result_dir + "/gen_model")

    # 生成モデル作成
    gen_model(params, result_dir)
