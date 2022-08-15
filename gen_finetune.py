import math
import os
import re
import argparse
from config import common_args, Parameters
from utils import setup_params, get_device, set_logging, dump_params
import logging
import pandas as pd
from typing import Any
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from bs4 import BeautifulSoup
import gc

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


def clean_txt(df) -> Any:
    # 不要タグの削除-クリーニング
    logger.info('Cleaning Test Dataset...')

    reg_obj = re.compile(r"<[^>]*?>")
    df = df.apply(lambda x: reg_obj.sub("", x).replace('\\u202f', '').replace('\\', ''))
    df = df.apply(lambda x: x.lstrip())
    # descriptions = descriptions.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())
    # reg_kanji = re.compile(u'[一-龥]')
    # reg_han = re.compile('[\u0000-\u007F]+')
    # df = df.apply(lambda x: reg_han.sub("", x))
    # df = df.apply(lambda x: x.encode('cp932', errors='ignore').decode('cp932'))  # cp932文字コードに含まれない文字は削除

    return df


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

    if not params.args['load_model']:
        print("non-load")

        tokenizer = GPT2Tokenizer.from_pretrained(params.gen_model_name, bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>', pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(params.gen_model_name, pad_token_id=tokenizer.eos_token_id)  # .cuda()
        model = model.to(params.device)
        model.resize_token_embeddings(len(tokenizer))

        # 読み込み
        descriptions = pd.read_csv(params.train_file_path)['description']

        # testデータ結合
        descriptions_test = pd.read_csv(params.test_file_path)['description']
        descriptions = pd.concat([descriptions, descriptions_test], axis=0, ignore_index=True).reset_index(drop=True).copy()

        # クリーニング
        descriptions = clean_txt(descriptions)
        descriptions.to_csv(result_dir + f'/dataset_clean.csv', index=False)

        max_length = max([len(tokenizer.encode(description)) for description in descriptions])

        # 学習データセット
        dataset = TrainDataset(descriptions, tokenizer, max_length=max_length)
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        gc.collect()

        logger.info('Training...')
        torch.cuda.empty_cache()
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        save_path = "./" + result_dir + "/gen_model"

        # 学習パラメータ report_to='none'
        training_args = TrainingArguments(output_dir=save_path, num_train_epochs=10, logging_steps=5000, save_steps=10000,
                                          per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                          warmup_steps=100, weight_decay=0.01, logging_dir=result_dir+"/",
                                          load_best_model_at_end=True,
                                          evaluation_strategy='epoch', save_strategy='epoch',
                                          save_total_limit=1,
                                          )

        # 学習
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                      'attention_mask': torch.stack([f[1] for f in data]),
                                                      'labels': torch.stack([f[0] for f in data])})
        trainer.train()

        # Save model
        trainer.save_model()

        torch.cuda.empty_cache()

        # 生成モデルからテキスト生成
        generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids.cuda()

        sample_outputs = model.generate(generated, do_sample=True, top_k=50, top_p=0.95,
                                        no_repeat_ngram_size=2,
                                        max_length=300, num_return_sequences=20)

        # サンプル生成テスト
        for i, sample_output in enumerate(sample_outputs):
            logger.info("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        logger.info("============================================")

        logger.info("Saving train_generated.csv...")
        # df_save.to_csv(result_dir + f'/train_generated.csv', index=False)
        logger.info("Saved train_generated.csv")

    else:
        logger.info("load_model" + params.args['load_model'])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # model path
        load_model_path = params.args['load_model']

        # モデル設定　
        tokenizer = GPT2Tokenizer.from_pretrained(params.gen_model_name, bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>', pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(load_model_path, pad_token_id=tokenizer.eos_token_id)
        model = model.to(params.device)
        model.resize_token_embeddings(len(tokenizer))

        torch.cuda.empty_cache()

        # 元データセットへの生成データの追加
        df = pd.read_csv(params.train_file_path)  # 読み込み

        label_num = params.num_classes  # ラベル番号
        index_num = df['id'].tail()

        # クリーニング
        df['description'] = clean_txt(df['description'])

        # 保存用データフレームにコピー
        df_save = df.copy()

        for i in range(1, label_num + 1):
            logger.info("============================================")
            logger.info("label: " + str(i))
            logger.info("============================================")
            logger.info("label: " + str(i))
            logger.info("============================================")

            # text MIN MAXの値が上手く動作していない．min9 max25
            logger.info(min(df[df['jobflag'] == i].description.str.len()))

            for n, text in enumerate(df[df['jobflag'] == i].description):
                df_size = df[df['jobflag'] == i].description.size  # 1job当たりのデータサイズ
                gen_num = math.ceil((params.sampling_num - df_size) / df_size)  # 生成回数
                txt_len_min = min(df[df['jobflag'] == i].description.str.len())
                txt_len_max = max(df[df['jobflag'] == i].description.str.len())
                df_txt_len = df[df['jobflag'] == i].description.str.len().median()  # 生成文字列数

                logger.info("text_len_MIN: " + str(txt_len_min) + " | text_len_MAX: " + str(txt_len_max))
                logger.info("============================================")

                # 先頭３単語抽出
                text = re.findall(r"[a-zA-Z]+", text)[0:3]
                map_text = map(str, text)
                text = " ".join(map_text)

                # ターゲットテキストの指定
                # target_text = text.split('.', 2)[0]  # + "."
                target_text = text

                # 生成準備
                generated = tokenizer(target_text, return_tensors="pt").input_ids.cuda()

                # テキスト生成設定
                # パラメータ:  num_beams=gen_num, no_repeat_ngram_size=2, early_stopping=True,
                sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                                min_length=25, max_length=800, top_p=0.95,
                                                num_return_sequences=gen_num)

            # text MIN MAXの値が上手く動作していない．min9 max25
            logger.info(min(df[df['jobflag'] == i].description.str.len()))

            for n, text in enumerate(df[df['jobflag'] == i].description):
                df_size = df[df['jobflag'] == i].description.size  # 1job当たりのデータサイズ
                gen_num = math.ceil((params.sampling_num - df_size) / df_size)  # 生成回数

                txt_len_min = min(df[df['jobflag'] == i].description.str.len())
                txt_len_max = max(df[df['jobflag'] == i].description.str.len())
                df_txt_len = df[df['jobflag'] == i].description.str.len().median()  # 生成文字列数

                logger.info("label: " + str(i) + "  description-num: " + str(n) + "/" + str(df_size))
                logger.info("txt_len_MIN: " + str(txt_len_min) + " | txt_len_MAX: " + str(txt_len_max))
                logger.info("============================================")

                # 先頭３単語抽出
                text = re.findall(r"[a-zA-Z]+", text)[0:3]
                map_text = map(str, text)
                target_text = " ".join(map_text)

                # 生成準備
                generated = tokenizer(target_text, return_tensors="pt").input_ids
                generated = generated.to(params.device)

                # テキスト生成設定
                # パラメータ:
                # num_beams=2, no_repeat_ngram_size=2, early_stopping=True,
                # temperature=0.7, repetition_penalty=1.0,
                # min_length=txt_len_min, max_length=txt_len_max,
                sample_outputs = model.generate(generated, do_sample=True, top_k=50, top_p=0.95,
                                                no_repeat_ngram_size=2,
                                                min_length=txt_len_min, max_length=txt_len_max,
                                                num_return_sequences=gen_num)

                # 生成データ
                for s, sample_output in enumerate(sample_outputs):
                    # データフレームに追加 strip("'").
                    list_add = [
                        [s + 1, tokenizer.decode(sample_output, skip_special_tokens=True).strip().replace('\n', " "),
                         i]]
                    # print(list_add)

                    df_add = pd.DataFrame(data=list_add, columns=['id', 'description', 'jobflag'])
                    # print(df_add)

                    logger.info("{}: {}".format(s, tokenizer.decode(sample_output, skip_special_tokens=True).strip()))

                    # Concat処理　元データセットの末尾に追加
                    logger.info("Dataframe Concat...")
                    df_save = pd.concat([df_save, df_add], axis=0, ignore_index=True).reset_index(drop=True)

                logger.info("============================================")

        logger.info("Saving train_generated.csv...")

        df_save.to_csv(result_dir + f'/train_generated.csv', index=False)

        logger.info("Saved train_generated.csv")


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
    os.mkdir(result_dir + "/gen_model")

    # ログ設定
    logger = logging.getLogger(__name__)
    set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定
    logger.info('parameters: ')
    logger.info(params)
    dump_params(params, f'{result_dir}')  # パラメータを出力

    # 生成モデル作成
    gen_model(params, result_dir)
