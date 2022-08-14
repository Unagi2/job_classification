"""
プロジェクト内のパラメータを管理するためのモジュール．

A) プログラムを書くときにやること．
  1) デフォルトパラメータを `Parameters` クラス内で定義する．
  2) コマンドライン引数を `common_args` 内で定義する．

B) パラメータを指定して実行するときにやること．
  1) `python config.py` とすると，デフォルトパラメータが `parameters.json` というファイルに書き出される．
  2) パラメータを指定する際は，Parametersクラスを書き換えるのではなく，jsonファイル内の値を書き換えて，
  `python train.py -p parameters.json`
  のようにjsonファイルを指定する．
"""

from dataclasses import dataclass, field
from email.policy import default
from utils import dump_params


@dataclass(frozen=True)
class Parameters:
    """
    プログラム全体を通して共通のパラメータを保持するクラス．
    ここにプロジェクト内で使うパラメータを一括管理する．
    """
    args: dict = field(default_factory=lambda: {})  # コマンドライン引数
    run_date: str = ''  # 実行時の時刻
    git_revision: str = ''  # 実行時のプログラムのGitのバージョン

    # dataset: str = 'Deu-mix'  # 使用するデータセットのID (4GLte-xxx/Ctw/Deu-xxx)
    # prediction_distance: int = 1  # どれだけ先の品質を推定するかを指定
    # preprocessed_data_path: str = None  # 前処理済みデータのパス

    device: str = 'cuda:0'  # デバイス

    # データセットパラメータ
    train_file_path: str = "./dataset/train.csv"
    test_file_path: str = "./dataset/test.csv"
    submit_sample_file_path: str = "./dataset/submit_sample.csv"
    gen_model_name: str = 'gpt2-large'
    ros: bool = True
    num_split: int = 5
    seed: int = 43
    sampling_num: int = 10000

    # BERT訓練データパラメータ
    models_dir: str = "/models/"
    model_name: str = 'allenai/scibert_scivocab_uncased'  # 候補は'bert-base-uncased', 'allenai/scibert_scivocab_uncased',
    train_batch_size: int = 32
    valid_batch_size: int = 128
    num_classes: int = 4
    epoch: int = 5
    # load_preprocessed_data: bool = True  # Trueなら処理済みファイルからロード
    # batch_size: int = 1  # ミニバッチ作成のためのバッチサイズ(1,2,4,8,16,・・・,1024,2048,4096）
    # data_length: float = float('inf')

    # param2: dict = field(default_factory=lambda: {'k1': 'v1', 'k2': 'v2'})  # リストや辞書で与える例


def common_args(parser):
    """
    コマンドライン引数を定義する関数．
    """
    parser.add_argument("-p", "--parameters", help="パラメータ設定ファイルのパスを指定．デフォルトはNone", type=str, default=None)
    parser.add_argument('-r', '--restart_lstm', type=int, default=0, help='n回目のLSTM学習からスタート')
    parser.add_argument('-l', '--load_model', default=None, type=str, help='ロードするモデルがあるディレクトリ')
    parser.add_argument('-s', '--save', type=str, default='result/', help='学習済みモデルを保存するディレクトリ')
    parser.add_argument('--roberta', action='store_true', help='学習済みモデルを保存するディレクトリ')
    # parser.add_argument("-a", "--arg1", type=int, help="arg1の説明", default=0)  # コマンドライン引数を指定
    # parser.add_argument("--prediction_distance", type=float, help="arg2の説明", default=1.0)  # コマンドライン引数を指定
    return parser


if __name__ == "__main__":
    dump_params(Parameters(), './', partial=True)  # デフォルトパラメータを出力
