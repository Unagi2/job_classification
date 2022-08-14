"""
データセットの準備など，前処理のモジュール
"""
import os
import argparse
import numpy as np
from config import common_args, Parameters
from utils import dump_params, setup_params, get_device
from utils import set_logging
import logging
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nlp
from bs4 import BeautifulSoup
from transformers import pipeline, set_seed
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def make_folded_df(params, csv_file, num_splits=5) -> Any:
    """データセット作成と前処理

        データセットのロードとリサンプリングを処理

        Args:
            params: パラメータ
            csv_file: ファイルパス
            num_splits: 分割数

        Returns:
            df:　データセット

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

    df = pd.read_csv(csv_file)  # ファイル読み込み

    # Class count オーバーサンプリング数指定用
    count_max = df["jobflag"].value_counts().max()
    # count_max = params.sampling_num

    # Divide by class
    df_class_1 = df[df['jobflag'] == 1]
    df_class_2 = df[df['jobflag'] == 2]
    df_class_3 = df[df['jobflag'] == 3]
    df_class_4 = df[df['jobflag'] == 4]

    # リサンプリング（ROS）
    df_class_1_over = df_class_1.sample(count_max, replace=True)
    df_class_2_over = df_class_2.sample(count_max, replace=True)
    df_class_3_over = df_class_3.sample(count_max, replace=True)
    # df_class_4_over = df_class_4.sample(count_max, replace=True)

    # Concat処理
    df_over = pd.concat([df_class_1_over, df_class_2_over, df_class_3_over, df_class_4], axis=0).reset_index()

    # print('Random over-sampling:')
    # print(df_over.jobflag.value_counts())

    # # 不要タグの削除-クリーニング
    df_over['description'] = df_over['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())

    df = df_over.drop('index', axis=1)
    df["jobflag"] = df["jobflag"] - 1
    df["kfold"] = np.nan
    df = df.rename(columns={'jobflag': 'labels'})
    label = df["labels"].tolist()

    skfold = StratifiedKFold(num_splits, shuffle=True, random_state=params.seed)
    for fold, (_, valid_indexes) in enumerate(skfold.split(range(len(label)), label)):
        for i in valid_indexes:
            df.iat[i, 3] = fold
    return df


def make_dataset(params, df, tokenizer) -> Any:
    """BERT用に処理できるようなデータ形式に変換

    Tokenizerを用いて文をIDに変換しデータセットを作成

    Args:
        params: パラメータ
        df: 変換前データセット
        tokenizer:

    Returns:
        datasets: 変換データセット

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

    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example["description"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=128))
    dataset.set_format(type='torch',
                       columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
                       device=params.device)
    return dataset

def make_dataset_roberta(params, df, tokenizer) -> Any:
    """RoBERTa用に処理できるようなデータ形式に変換

    Tokenizerを用いて文をIDに変換しデータセットを作成

    Args:
        params: パラメータ
        df: 変換前データセット
        tokenizer:

    Returns:
        datasets: 変換データセット

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

    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example["description"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=128))
    dataset.set_format(type='torch',
                       columns=['input_ids', 'attention_mask', 'labels'],
                       device=params.device)
    return dataset


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

    # trainデータセット作成
    df = make_folded_df(params, params.train_file_path, params.num_split)

    for fold in range(params.num_split):
        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)
