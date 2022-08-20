"""
機械学習モデルを用いないData Augmentationを行うモジュール.
"""
import os
import argparse
from config import common_args, Parameters
from utils import dump_params, setup_params, get_device, seed_everything
from utils import set_logging
from gen_finetune import clean_txt
import logging
import re

import pandas as pd
from textaugment import EDA
import nltk

def easy_data_augment(params, result_dir):
    """
    Easy Data Augmentionという手法でデータの水増しを行う関数.
    Args:
        params: config内のパラメータ群
        result_dir: 保存先ディレクトリ
    """
    logger.info('Data augment...')
    seed_everything(params.seed)
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    eda_ins = EDA(random_state=params.seed)

    data_original = params.train_eda_file_path
    data_augmented = result_dir + '/' + data_original.split('/')[-1].removesuffix('.csv') + '_augmented.csv'

    df = pd.read_csv(data_original, index_col=0)
    df['description'] = clean_txt(df['description'])

    descriptions = []
    jobflags = []
    logger.info("Data Augmenting...")
    for description, jobflag in zip(df['description'],df['jobflag']):
        n = round(count_words(description.replace('e.g.', 'for example')) * params.alpha_xx)
        for i in range(params.num_aug):
            descriptions.append(eda(eda_ins, description, n, params.alpha_xx))
            jobflags.append(jobflag)
    
    df_augmented = pd.DataFrame(data={'description': descriptions, 'jobflag': jobflags}, columns=df.columns)
    df_augmented.index.name = df.index.name
    df_concat = pd.concat([df,df_augmented], ignore_index=True)
    df_concat.to_csv(data_augmented, index_label=df.index.name)
    logger.info("Data Augmentation Complete!")


def count_words(sentence):
    words = re.split(r'\s|\,|\.|\(|\)', sentence)
    words = [word for word in words if word != '']
    return len(words)

def eda(eda_ins, sentence, n, p=0.02):
    """
    Easy Data Augmentionを行う関数.
    Args:
        sentence: 処理を行う文章
        n: 文章中の加工を行う単語数
        p: 文章中の単語が加工される確率
    """
    augmented_sentence = eda_ins.synonym_replacement(sentence, n)
    augmented_sentence = eda_ins.random_insertion(augmented_sentence, n)
    augmented_sentence = eda_ins.random_swap(augmented_sentence, n)
    augmented_sentence = eda_ins.random_deletion(augmented_sentence, p)
    return augmented_sentence


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
    result_dir = f'result/{params.run_date}_data_augmentation'  # 結果出力ディレクトリ
    os.mkdir(result_dir)  # 実行日時を名前+'data_augmentation'とするディレクトリを作成
    dump_params(params, f'{result_dir}')  # パラメータを出力

    logger = logging.getLogger(__name__)
    set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定

    logger.info('parameters: ')
    logger.info(params)

    easy_data_augment(params, result_dir)