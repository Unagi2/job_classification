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
│   └── test.csv
├── config.py           # パラメータ定義
├── main.py             # 実行ファイル
├── parameters.json     # パラメータ指定用ファイル
├── result              # 結果出力ディレクトリ
│   └── 20211026_165841
├── preprocess.py       # 前処理
├── model.py            # 学習
├── predict.py          # 予測
└── utils.py            # 共有関数群
```
