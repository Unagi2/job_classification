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

logger = logging.getLogger(__name__)


def get_train_data(params) -> list[Any]:
    """学習データのロード

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

    get_train_data(params)
