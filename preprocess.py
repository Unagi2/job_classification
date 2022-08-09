"""
データセットの準備など，前処理のモジュール
"""
import os
import argparse
from config import common_args, Parameters
from utils import dump_params, setup_params
from utils import set_logging
import logging
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def get_train_data(params) -> list[Any]:
    """学習データのロード

    データセットのロードとリサンプリングを処理

    Args:
        params: パラメータを含む辞書

    Returns:
        data_list(list): データ
        label_list(list): ラベル

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    logger.info('Loading Train Dataset...')

    # 学習用データの読み込み
    df_train = pd.read_csv("./dataset/train.csv")

    data_list = df_train["description"]
    label_list = df_train["jobflag"]

    # Class count オーバーサンプリング数指定用
    count_max = df_train["jobflag"].value_counts().max()

    # Divide by class
    df_class_1 = df_train[df_train['jobflag'] == 1]
    df_class_2 = df_train[df_train['jobflag'] == 2]
    df_class_3 = df_train[df_train['jobflag'] == 3]
    df_class_4 = df_train[df_train['jobflag'] == 4]

    # リサンプリング（ROS）
    df_class_1_over = df_class_1.sample(count_max, replace=True)
    df_class_2_over = df_class_2.sample(count_max, replace=True)
    df_class_3_over = df_class_3.sample(count_max, replace=True)

    # Concat処理
    df_train_over = pd.concat([df_class_4, df_class_1_over, df_class_2_over, df_class_3_over], axis=0).reset_index()

    # print('Random over-sampling:')
    # print(df_train_over.jobflag.value_counts())

    df_train_crean = []
    for i in range(0, len(df_train_over['description'])):
        df_train_crean.append(BeautifulSoup(df_train_over['description'][i]).get_text())

    data_list = pd.Series(df_train_crean)
    label_list = df_train_over['jobflag']

    return data_list, label_list


def get_test_data(params) -> list[Any]:
    """評価データのロード

    データセットのロードとリサンプリングを処理

    Args:
        params: パラメータ
    Returns:
        data_list: データ
        label_list: ラベル

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    logger.info('Loading Train Dataset...')

    data_list = []
    label_list = []

    # 学習用データの読み込み
    df_test = pd.read_csv("./dataset/train.csv")

    return data_list, label_list


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)  # コマンドライン引数引数を読み込み
    # parser.add_argument("--main")  # 実行スクリプト固有のコマンドライン引数があればここに記入する．
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

    train_data, train_label = get_train_data(params)
    # test_data, test_label = get_test_data(params)
