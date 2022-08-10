"""
学習モデルを用いて推論を行う
"""
import os
import logging
import argparse
from config import common_args, Parameters
from model_BERT import Classifier
from preprocess import make_dataset
from utils import dump_params, setup_params, get_device
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer
from bs4 import BeautifulSoup

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
    for fold in range(params.num_split):
        model = Classifier(params.model_name)
        model.load_state_dict(torch.load("./" + result_dir + "/" + params.models_dir + f"best_{params.model_name}_{fold}.pth"))
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
    os.mkdir(result_dir + "/output")
    dump_params(params, f'{result_dir}')  # パラメータを出力

    cv = 0
    predict(params, cv)
