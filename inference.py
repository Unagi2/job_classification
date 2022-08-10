"""
学習モデルを用いて推論を行う
"""
from genericpath import isdir
import os
from os.path import isdir
import logging
import argparse
from os.path import isdir

from config import common_args, Parameters
from model_BERT import Classifier
from model_BERT_Conv import Classifier_Conv
from model_RoBERTa import Classifier_RoBERTa
from preprocess import make_dataset, make_dataset_roberta
from utils import dump_params, setup_params, get_device
from utils import set_logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer
from bs4 import BeautifulSoup
import shutil

logger = logging.getLogger(__name__)


def predict(params, cv, result_dir) -> None:
    """推論の実行

    学習モデルから推論を行う

    Args:
        params: パラメータ
        cv: 平均f1スコア
        result_dir: 保存親ディレクトリ

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
    logger.info('inference test data...')

    models = []

    if '/' in params.model_name:
        model_name_dir = params.model_name.split('/')[1]  # model_nameに/が含まれていることがあるため、/以降のみを使う
    else:
        model_name_dir = params.model_name

    for fold in range(params.num_split):
        if params.use_cnn:
            model = Classifier_Conv(params.model_name, num_classes=params.num_classes)
        else:
            model = Classifier(params.model_name, num_classes=params.num_classes)
        model.load_state_dict(torch.load("./" + result_dir + "/" + params.models_dir + f"best_{model_name_dir}_{fold}.pth"))
        model.to(params.device)
        model.eval()
        models.append(model)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    # テストデータ読み込み
    test_df = pd.read_csv(params.test_file_path)

    # 不要タグの削除-クリーニング
    test_df['description'] = test_df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())
    # test_df = test_df.drop('index', axis=1)

    test_df["labels"] = -1
    test_dataset = make_dataset(params, test_df, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.valid_batch_size, shuffle=False)

    with torch.no_grad():
        progress = tqdm(test_dataloader, total=len(test_dataloader))
        final_output = []

        for batch in progress:
            progress.set_description("<Test>")

            attention_mask, input_ids, labels, token_type_ids = batch.values()

            outputs = []
            for model in models:
                output = model(input_ids, attention_mask, token_type_ids)
                outputs.append(output)

            outputs = sum(outputs) / len(outputs)
            outputs = torch.softmax(outputs, dim=1).cpu().detach().tolist()
            outputs = np.argmax(outputs, axis=1)

            final_output.extend(outputs)

    submit = pd.read_csv(os.path.join(params.submit_sample_file_path), names=["id", "labels"])
    submit["labels"] = final_output
    submit["labels"] = submit["labels"] + 1
    try:
        submit.to_csv("./" + result_dir + "/output/submission_cv{}.csv".format(str(cv).replace(".", "")[:10]), index=False, header=False)
    except NameError:
        submit.to_csv("./" + result_dir + "/output/submission.csv", index=False, header=False)

    logger.info('Inference complete!')

def predict_roberta(params, cv, result_dir) -> None:
    """推論の実行

    学習モデルから推論を行う

    Args:
        params: パラメータ
        cv: 平均f1スコア
        result_dir: 保存親ディレクトリ

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
    logger.info('inference test data...')

    models = []

    if '/' in params.model_name:
        model_name_dir = params.model_name.split('/')[1] # model_nameに/が含まれていることがあるため、/以降のみを使う
    else:
        model_name_dir = params.model_name

    for fold in range(params.num_split):
        model = Classifier_RoBERTa(params.model_name)
        model.load_state_dict(torch.load("./" + result_dir + "/" + params.models_dir + f"best_{model_name_dir}_{fold}.pth"))
        model.to(params.device)
        model.eval()
        models.append(model)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    # テストデータ読み込み
    test_df = pd.read_csv(params.test_file_path)

    # 不要タグの削除-クリーニング
    test_df['description'] = test_df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lstrip())
    # test_df = test_df.drop('index', axis=1)

    test_df["labels"] = -1
    test_dataset = make_dataset_roberta(params, test_df, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.valid_batch_size, shuffle=False)

    with torch.no_grad():
        progress = tqdm(test_dataloader, total=len(test_dataloader))
        final_output = []

        for batch in progress:
            progress.set_description("<Test>")

            attention_mask, input_ids, labels = batch.values()

            outputs = []
            for model in models:
                output = model(input_ids, attention_mask)
                outputs.append(output)

            outputs = sum(outputs) / len(outputs)
            outputs = torch.softmax(outputs, dim=1).cpu().detach().tolist()
            outputs = np.argmax(outputs, axis=1)

            final_output.extend(outputs)

    submit = pd.read_csv(os.path.join(params.submit_sample_file_path), names=["id", "labels"])
    submit["labels"] = final_output
    submit["labels"] = submit["labels"] + 1
    try:
        submit.to_csv("./" + result_dir + "/output/submission_cv{}.csv".format(str(cv).replace(".", "")[:10]), index=False, header=False)
    except NameError:
        submit.to_csv("./" + result_dir + "/output/submission.csv", index=False, header=False)

    logger.info('Inference complete!')


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
    result_dir = params.args['load_model']  # 結果出力ディレクトリ
    # os.mkdir(result_dir)  # 実行日時を名前とするディレクトリを作成
    if isdir(result_dir + "/output"):
        shutil.rmtree(result_dir + "/output")
    os.mkdir(result_dir + "/output")
    if isdir(result_dir + "/inference"):
        shutil.rmtree(result_dir + "/inference")
    os.mkdir(result_dir + "/inference")
    dump_params(params, f'{result_dir}' + "/inference")  # パラメータを出力

    cv = 0

    if '/' in params.model_name:
        model_name_dir = params.model_name.split('/')[1]  # model_nameに/が含まれていることがあるため、/以降のみを使う
    else:
        model_name_dir = params.model_name

    with open(f"./{result_dir}/{model_name_dir}_result.txt", mode='r') as f:
        lines = f.readlines()
        s_lines = [line.strip() for line in lines]
        for line in s_lines:
            if 'CV' in line:
                cv = float(line.split(':')[1].strip())
    
    # ログ設定
    logger = logging.getLogger(__name__)
    set_logging(result_dir + "/inference")  # ログを標準出力とファイルに出力するよう設定

    logger.info('parameters: ')
    logger.info(params)

    if params.args['roberta']:
        predict_roberta(params, cv, result_dir)
    else:
        predict(params, cv, result_dir)
