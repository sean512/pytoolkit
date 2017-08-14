"""DeepLearning(主にKeras)関連。

kerasをimportしてしまうとTensorFlowの初期化が始まって重いので、
importしただけではkerasがimportされないように作っている。

"""
import csv
import pathlib

import numpy as np


def my_callback_factory():
    """色々入りのKerasのCallbackクラスを作って返す。

    TerminateOnNaN+ReduceLROnPlateau+EarlyStopping+CSVLoggerのようなもの。

    lossを監視して学習率を制御して、十分学習できたら終了する。
    ついでにログも出す。(ミニバッチ単位＆エポック単位)

    # 引数

    - log_dir: ログ出力先
    - batch_log_name: ミニバッチ単位のログファイル名
    - epoch_log_name: エポック単位のログファイル名

    - lr_list: epoch毎の学習率のリスト (base_lrと排他)

    - base_lr: 学習率の自動調整のベースとする学習率 (lr_listと排他)
    - max_reduces: 学習率の自動調整時、最大で何回学習率を減らすのか
    - reduce_factor: 学習率の自動調整時、学習率を減らしていく割合
    - beta1: lossを監視する際の指数移動平均の係数
    - beta2: lossを監視する際の指数移動平均の係数
    - margin_iterations: 学習率の自動調整時、このバッチ数分までは誤差を考慮して学習率を下げない

    - verbose: 学習率などをprintするなら1

    """
    import keras
    import keras.backend as K

    class _MyCallback(keras.callbacks.Callback):

        def __init__(self, log_dir='.',
                     lr_list=None,
                     base_lr=None,
                     verbose=1,
                     batch_log_name='batchlog.tsv',
                     epoch_log_name='epochlog.tsv',
                     max_reduces=6, reduce_factor=1 / np.sqrt(10),
                     beta1=0.998, beta2=0.999, margin_iterations=100):
            super().__init__()
            # 設定
            assert (lr_list is None) != (base_lr is None)  # どちらか片方のみ必須
            self.log_dir = log_dir
            self.batch_log_name = batch_log_name
            self.epoch_log_name = epoch_log_name
            self.lr_list = lr_list
            self.base_lr = base_lr
            self.max_reduces = max_reduces
            self.reduce_factor = reduce_factor
            self.beta1 = beta1
            self.beta2 = beta2
            self.margin_iterations = margin_iterations
            self.verbose = verbose
            # あとで使うものたち
            self.batch_log_file = None
            self.epoch_log_file = None
            self.batch_log_writer = None
            self.epoch_log_writer = None
            self.keys = None
            self.iterations = 0
            self.iterations_per_reduce = 0
            self.ema1 = 0
            self.ema2 = 0
            self.reduces = 0
            self.stopped_epoch = 0
            self.epoch = 0
            self.reduce_on_epoch_end = False

        def on_train_begin(self, logs=None):
            # ログファイル作成
            d = pathlib.Path(self.log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.batch_log_file = d.joinpath(self.batch_log_name).open('w')
            self.epoch_log_file = d.joinpath(self.epoch_log_name).open('w')
            self.batch_log_writer = csv.writer(self.batch_log_file, delimiter='\t')
            self.epoch_log_writer = csv.writer(self.epoch_log_file, delimiter='\t')
            self.batch_log_writer.writerow(['epoch', 'batch', 'loss', 'delta_ema'])
            self.keys = None
            # 学習率の設定(base_lr)
            if self.base_lr is not None:
                K.set_value(self.model.optimizer.lr, float(self.base_lr))
                if self.verbose >= 1:
                    print('lr = {}'.format(float(K.get_value(self.model.optimizer.lr))))
            # 色々初期化
            self.iterations = 0
            self.iterations_per_reduce = 0
            self.ema1 = 0
            self.ema2 = 0
            self.reduces = 0
            self.stopped_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            if self.lr_list is not None:
                # 学習率の設定(lr_list)
                lr = self.lr_list[epoch]
                if self.verbose >= 1:
                    if epoch == 0 or lr != self.lr_list[epoch - 1]:
                        print('lr = {}'.format(float(lr)))
                K.set_value(self.model.optimizer.lr, float(lr))
            elif self.reduce_on_epoch_end:
                if self.verbose >= 1:
                    print('lr = {}'.format(float(K.get_value(self.model.optimizer.lr))))
            # 色々初期化
            self.epoch = epoch
            self.reduce_on_epoch_end = False

        def on_batch_begin(self, batch, logs=None):
            pass

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')

            # nanチェック(一応)
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    print('Batch %d: Invalid loss, terminating training' % (batch))
                    self.model.stop_training = True

            # lossの指数移動平均の算出
            self.ema1 = loss * (1 - self.beta1) + self.ema1 * self.beta1
            self.ema2 = loss * (1 - self.beta2) + self.ema2 * self.beta2
            # Adam風補正
            self.iterations += 1
            hm1 = self.ema1 / (1 - self.beta1 ** self.iterations)
            hm2 = self.ema2 / (1 - self.beta2 ** self.iterations)
            delta_ema = hm2 - hm1
            if self.base_lr is not None:
                # lossの減少が止まってそうなら次のepochから学習率を減らす。
                self.iterations_per_reduce += 1
                if delta_ema <= 0 and self.margin_iterations <= self.iterations_per_reduce:
                    self.reduce_on_epoch_end = True

            # batchログ出力
            self.batch_log_writer.writerow([self.epoch + 1, batch + 1, loss, delta_ema])

        def on_epoch_end(self, epoch, logs=None):
            # batchログ出力
            self.batch_log_file.flush()
            # epochログ出力
            if not self.keys:
                self.keys = sorted(logs.keys())
                self.epoch_log_writer.writerow(['epoch', 'lr'] + self.keys)
            lr = K.get_value(self.model.optimizer.lr)
            metrics = [logs.get(k) for k in self.keys]
            self.epoch_log_writer.writerow([epoch + 1, lr] + metrics)
            self.epoch_log_file.flush()
            # 学習率を減らす/限界まで下がっていたら学習終了
            if self.reduce_on_epoch_end:
                if self.max_reduces <= self.reduces:
                    # 限界まで下がっていたら学習終了
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                else:
                    # 学習率を減らす
                    self.reduces += 1
                    lr = self.base_lr * self.reduce_factor ** self.reduces
                    K.set_value(self.model.optimizer.lr, float(lr))
                    self.iterations_per_reduce = 0  # 安全装置のリセット

        def on_train_end(self, logs=None):
            self.batch_log_file.close()
            self.epoch_log_file.close()

    return _MyCallback


def session(config=None, gpu_options=None):
    """TensorFlowのセッションの初期化・後始末。

    # 使い方

    ```
    with tk.dl.session():

        # kerasの処理

    ```

    """
    import keras.backend as K

    class _Scope(object):  # pylint: disable=R0903

        def __init__(self, config=None, gpu_options=None):
            self.config = config or {}
            self.gpu_options = gpu_options or {}

        def __enter__(self):
            if K.backend() == 'tensorflow':
                import tensorflow as tf
                self.config.update({'allow_soft_placement': True})
                self.gpu_options.update({'allow_growth': True})
                K.set_session(
                    tf.Session(
                        config=tf.ConfigProto(
                            **self.config, gpu_options=tf.GPUOptions(**self.gpu_options))))

        def __exit__(self, *exc_info):
            if K.backend() == 'tensorflow':
                K.clear_session()

    return _Scope(config=config, gpu_options=gpu_options)
