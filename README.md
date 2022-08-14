# Job-classification

Signateによる職種分類コンペのプロジェクト


## Requirement
- Python 3.9


## Installation
- 結果出力用ディレクトリを作成
```shell
mkdir dataset
mkdir result
```
- 各種モジュールのインストール
```shell
pip install -r requirements.txt
```

### Dataset Installation
コンペサイトのデータから，trainとtestデータをダウンロードする．
- [このサイト](https://signate.jp/competitions/735/data) から 各csvダウンロード

- `./dataset/`下に保存

## Usage

### メインプログラムを実行
- メインプログラムを実行．
  - `result/[日付][実行時刻]/` 下に実行結果とログが出力されます．
```shell
python main.py
```
- デフォルトのパラメータ設定をjson出力．
```shell
python config.py  # parameters.jsonというファイルが出力される．
```
- 以下のように，上記で生成されるjsonファイルの数値を書き換えて，実行時のパラメータを指定できます．
```shell
python -p parameters.json main.py
```
- 詳しいコマンドの使い方は以下のように確認できます．
```shell
python main.py -h
```

## Preprocess
- 前処理のみ実行
python preprocess.py

## Training
- 学習及び予測の実行
nohup python train_BERT.py &  # BERT系
nohup python train_RoBERTa.py &  # RoBERTa系

## Inference
- 学習済みのモデルを用いて予測を実行
nohup python.py inference.py --load_model result/{結果ディレクトリ名}

## Parameter Settings

- 指定できるパラメータは以下の通り．
```json
{

}
```

## Directory Structure
- プロジェクトの構成は以下の通り．
```shell
.
├── dataset             # データセット
│   ├── train.csv
│   ├── test.csv
│   └── submit_sample.csv
├── config.py           # パラメータ定義
├── parameters.json     # パラメータ指定用ファイル
├── requirements.txt    # パッケージ情報
├── result              # 結果出力ディレクトリ
│   └── 20211026_165841
├── preprocess.py       # 前処理
├── model_BERT.py       # BERTモデル
├── train_BERT.py       # BERT学習(実行ファイル)
├── inference.py        # 推論
└── utils.py            # 共有関数群
```
