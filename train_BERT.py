"""
実行ファイル
"""
import os
import argparse
from config import common_args, Parameters
from inference import predict
from model_BERT import Classifier
from preprocess import make_dataset, make_folded_df
from utils import dump_params, setup_params, get_device, seed_everything
from utils import set_logging
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# training function
def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    """学習に関する損失や各種スコア

    学習の損失計算及び最適化

    Args:
        dataloader:
        model:
        criterion:
        optimizer:
        scheduler:
        device:　GPU
        epoch: エポック数

    Returns:
        train_loss: モデルによる予測と正解との誤差
        train_acc: 正解率
        train_f1: コンペの評価指標

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """

    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch + 1}")

        attention_mask, input_ids, labels, token_type_ids = batch.values()
        del batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids)
        del input_ids, attention_mask, token_type_ids
        loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        del outputs

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss
        total_corrects += torch.sum(preds == labels)

        all_labels += labels.tolist()
        all_preds += preds.tolist()
        del labels, preds

        progress.set_postfix(loss=total_loss / (i + 1), f1=f1_score(all_labels, all_preds, average="macro"))

    train_loss = total_loss / len(dataloader)
    train_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    return train_loss, train_acc, train_f1


def eval_fn(dataloader, model, criterion, device, epoch):
    """検証データに関する損失や各種スコア

    検証データの損失計算及び最適化

    Args:
        dataloader:
        model:
        criterion:
        device:　GPU
        epoch: エポック数

    Returns:
        valid_loss: モデルによる予測と正解との誤差
        valid_acc: 正解率
        valid_f1: コンペの評価指標

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))

        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch + 1}")

            attention_mask, input_ids, labels, token_type_ids = batch.values()
            del batch

            outputs = model(input_ids, attention_mask, token_type_ids)
            del input_ids, attention_mask, token_type_ids
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            del outputs

            total_loss += loss.item()
            del loss
            total_corrects += torch.sum(preds == labels)

            all_labels += labels.tolist()
            all_preds += preds.tolist()
            del labels, preds

            progress.set_postfix(loss=total_loss / (i + 1), f1=f1_score(all_labels, all_preds, average="macro"))

    valid_loss = total_loss / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = f1_score(all_labels, all_preds, average="macro")

    return valid_loss, valid_acc, valid_f1


def plot_training(train_losses, train_accs, train_f1s,
                  valid_losses, valid_accs, valid_f1s,
                  epoch, fold, result_dir):
    """結果の可視化

    ３つの評価指標をもとに可視化

    Args:
        train_losses:　モデルによる予測と正解との誤差
        train_accs:　正解率
        train_f1s:　コンペの評価指標
        valid_losses:　モデルによる予測と正解との誤差
        valid_accs:　正解率
        valid_f1s:　コンペの評価指標
        epoch:　エポック数
        fold:　k-fold値
        result_dir: 保存親ディレクトリ

    Returns:
        valid_loss: モデルによる予測と正解との誤差
        valid_acc: 正解率
        valid_f1: コンペの評価指標

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """

    loss_df = pd.DataFrame({"Train": train_losses,
                            "Valid": valid_losses},
                           index=range(1, epoch + 2))
    loss_ax = sns.lineplot(data=loss_df).get_figure()
    loss_ax.savefig(f"./" + result_dir + f"/figures/loss_plot_fold={fold}.png", dpi=300)
    loss_ax.clf()

    acc_df = pd.DataFrame({"Train": train_accs,
                           "Valid": valid_accs},
                          index=range(1, epoch + 2))
    acc_ax = sns.lineplot(data=acc_df).get_figure()
    acc_ax.savefig(f"./" + result_dir + f"/figures/acc_plot_fold={fold}.png", dpi=300)
    acc_ax.clf()

    f1_df = pd.DataFrame({"Train": train_f1s,
                          "Valid": valid_f1s},
                         index=range(1, epoch + 2))
    f1_ax = sns.lineplot(data=f1_df).get_figure()
    f1_ax.savefig(f"./" + result_dir + f"/figures/f1_plot_fold={fold}.png", dpi=300)
    f1_ax.clf()


def trainer(params, fold, df, result_dir):
    """学習関数

    学習を行うため他関数をまとめる

    Args:
        params:　モデルによる予測と正解との誤差
        fold:　k-fold値
        df: データセット
        result_dir: 保存親ディレクトリ

    Returns:
        best_f1: 最高f1スコア

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    train_dataset = make_dataset(params, train_df, tokenizer)
    valid_dataset = make_dataset(params, valid_df, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.train_batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=params.valid_batch_size, shuffle=False
    )

    model = Classifier(params.model_name, num_classes=params.num_classes)
    model = model.to(params.device)

    if '/' in params.model_name:
        model_name_dir = params.model_name.split('/')[1]  # model_nameに/が含まれていることがあるため、/以降のみを使う
    else:
        model_name_dir = params.model_name

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
    # ダミーのスケジューラー

    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0

    for epoch in range(params.epoch):
        train_loss, train_acc, train_f1 = train_fn(train_dataloader, model, criterion, optimizer, scheduler, params.device,
                                                   epoch)
        valid_loss, valid_acc, valid_f1 = eval_fn(valid_dataloader, model, criterion, params.device, epoch)
        logger.info(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        plot_training(train_losses, train_accs, train_f1s,
                      valid_losses, valid_accs, valid_f1s,
                      epoch, fold, result_dir)

        best_loss = valid_loss if valid_loss < best_loss else best_loss
        besl_acc = valid_acc if valid_acc > best_acc else best_acc
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            logger.info("model saving!")
            torch.save(model.state_dict(), "./" + result_dir + params.models_dir + f"best_{model_name_dir}_{fold}.pth")

        # logger.info("\n")

    return best_f1


def train(params, result_dir):
    """学習実行

    BERT学習を実行

    Args:
        params:　モデルによる予測と正解との誤差
        result_dir: 保存親ディレクトリ

    Returns:
        cv: 平均f1スコア

    Examples:
        関数の使い方

    Raises:
        例外の名前: 例外の説明

    Yields:
        戻り値の型: 戻り値についての説明

    Note:
        注意事項
    """
    logger.info('Train...')

    seed_everything(params.seed)

    # training
    df = make_folded_df(params, params.train_file_path, params.num_split)
    f1_scores = []
    for fold in range(params.num_split):
        print(f"fold {fold}", "=" * 80)
        f1 = trainer(params, fold, df, result_dir)
        f1_scores.append(f1)
        logger.info(f"<fold={fold}> best score: {f1}\n")

    cv = sum(f1_scores) / len(f1_scores)
    logger.info(f"CV: {cv}")

    lines = ""
    for i, f1 in enumerate(f1_scores):
        line = f"fold={i}: {f1}\n"
        lines += line
    lines += f"CV    : {cv}"

    if '/' in params.model_name:
        model_name_dir = params.model_name.split('/')[1]  # model_nameに/が含まれていることがあるため、/以降のみを使う
    else:
        model_name_dir = params.model_name

    with open(f"./{result_dir}/{model_name_dir}_result.txt", mode='w') as f:
        f.write(lines)

    return cv


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
    result_dir = f'result/{params.run_date}'  # 結果出力ディレクトリ

    os.mkdir(result_dir)  # 実行日時を名前とするディレクトリを作成
    os.mkdir(result_dir + "/figures")
    os.mkdir(result_dir + "/models")
    os.mkdir(result_dir + "/output")
    dump_params(params, f'{result_dir}')  # パラメータを出力

    # ログ設定
    logger = logging.getLogger(__name__)
    set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定

    logger.info('parameters: ')
    logger.info(params)

    # 学習
    cv = train(params, result_dir)

    # 推論
    predict(params, cv, result_dir)
