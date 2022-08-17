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
from sched import scheduler
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
    train_file_path: str = "./dataset/train_augmented.csv"
    train_gen_file_path: str = "./dataset/train_augmented.csv"
    test_file_path: str = "./dataset/test.csv"
    submit_sample_file_path: str = "./dataset/submit_sample.csv"
    gen_model_name: str = 'distilgpt2'
    ros: bool = False  # オーバーサンプリングによってデータセットを増やすかどうか(Falseの場合生成済みファイルから読み込む)
    num_split: int = 5
    seed: int = 45
    sampling_num: int = 4000

    # BERT訓練データパラメータ
    lr = 2e-5 # 学習率
    gamma = 0.8  # スケジューラーの更新率. 1epochごとに学習率に乗算される.

    models_dir: str = "/models/"
    model_name: str =  'allenai/scibert_scivocab_uncased'
    # 候補は'bert-base-uncased', 'allenai/scibert_scivocab_uncased', 'roberta-base'
    # model_name_for_roberta: str = '' 
    train_batch_size: int = 32
    valid_batch_size: int = 128
    num_classes: int = 4
    epoch: int = 5
    use_cnn: bool = True  # BERTの最終層の後、1次元のConvolutionalネットワークを通すかどうか
    # load_preprocessed_data: bool = True  # Trueなら処理済みファイルからロード
    # batch_size: int = 1  # ミニバッチ作成のためのバッチサイズ(1,2,4,8,16,・・・,1024,2048,4096）
    # data_length: float = float('inf')
    # param2: dict = field(default_factory=lambda: {'k1': 'v1', 'k2': 'v2'})  # リストや辞書で与える例

    # Data Augmentationで与えるパラメータ
    alpha_xx: float = 0.05  # 元の文章中の単語が加工される確率
    num_aug: int = 8  # 一つの文章から生成される類似文章数


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
