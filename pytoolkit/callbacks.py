"""DeepLearning(主にKeras)関連。"""
import csv
import pathlib
import time

import numpy as np
import sklearn

import pytoolkit as tk
from . import K, keras


class LearningRateStepDecay(keras.callbacks.Callback):
    """よくある150epoch目と225epoch目に学習率を1/10するコールバック。"""
    
    def __init__(self, reduce_epoch_rates=(0.5, 0.75), factor=0.1, epochs=None):
        super().__init__()
        self.reduce_epoch_rates = reduce_epoch_rates
        self.factor = factor
        self.epochs = epochs
        self.start_lr = None
        self.reduce_epochs = None
    
    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.lr))
        epochs = self.epochs or self.params["epochs"]
        self.reduce_epochs = [
            min(max(int(epochs * r), 1), epochs) for r in self.reduce_epoch_rates
        ]
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 in self.reduce_epochs:
            lr1 = K.get_value(self.model.optimizer.lr)
            lr2 = lr1 * self.factor
            K.set_value(self.model.optimizer.lr, lr2)
            tk.log.get(__name__).info(
                f"Epoch {epoch + 1}: Learning rate {lr1:.1e} -> {lr2:.1e}"
            )
    
    def on_train_end(self, logs=None):
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.lr, self.start_lr)


class CosineAnnealing(keras.callbacks.Callback):
    """Cosine Annealing without restart。

    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>

    """
    
    def __init__(self, factor=0.01, epochs=None, warmup_epochs=5):
        assert factor < 1
        super().__init__()
        self.factor = factor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.start_lr = None
    
    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.lr))
    
    def on_epoch_begin(self, epoch, logs=None):
        lr_max = self.start_lr
        lr_min = self.start_lr * self.factor
        if epoch + 1 < self.warmup_epochs:
            lr = lr_max * (epoch + 1) / self.warmup_epochs
        else:
            r = (epoch + 1) / (self.epochs or self.params["epochs"])
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
        K.set_value(self.model.optimizer.lr, float(lr))
    
    def on_train_end(self, logs=None):
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.lr, self.start_lr)


class TSVLogger(keras.callbacks.Callback):
    """ログを保存するコールバック。Horovod使用時はrank() == 0のみ有効。

    Args:
        filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
        append: 追記するのか否か。

    """
    
    def __init__(self, filename, append=False, enabled=None):
        super().__init__()
        self.filename = pathlib.Path(filename)
        self.append = append
        self.enabled = enabled if enabled is not None else tk.hvd.is_master()
        self.log_file = None
        self.log_writer = None
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        if self.enabled:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.filename.open(
                "a" if self.append else "w", buffering=65536
            )
            self.log_writer = csv.writer(
                self.log_file, delimiter="\t", lineterminator="\n"
            )
            self.log_writer.writerow(
                ["epoch", "lr"] + self.params["metrics"] + ["time"]
            )
        else:
            self.log_file = None
            self.log_writer = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        assert self.epoch_start_time is not None
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)
        elapsed_time = time.time() - self.epoch_start_time
        
        def _format_metric(logs, k):
            value = logs.get(k)
            if value is None:
                return "<none>"
            return f"{value:.4f}"
        
        metrics = [_format_metric(logs, k) for k in self.params["metrics"]]
        row = (
            [epoch + 1, format(logs["lr"], ".1e")]
            + metrics
            + [str(int(np.ceil(elapsed_time)))]
        )
        if self.log_file is not None:
            self.log_writer.writerow(row)
            self.log_file.flush()
    
    def on_train_end(self, logs=None):
        if self.log_file is not None:
            self.log_file.close()
        self.append = True  # 同じインスタンスの再利用時は自動的に追記にする


class EpochLogger(keras.callbacks.Callback):
    """DEBUGログを色々出力するcallback。Horovod使用時はrank() == 0のみ有効。"""
    
    def __init__(self, enabled=None):
        super().__init__()
        self.enabled = enabled if enabled is not None else tk.hvd.is_master()
        self.train_start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        assert self.train_start_time is not None
        assert self.epoch_start_time is not None
        lr = K.get_value(self.model.optimizer.lr)
        now = time.time()
        elapsed_time = now - self.epoch_start_time
        time_per_epoch = (now - self.train_start_time) / (epoch + 1)
        eta = time_per_epoch * (self.params["epochs"] - epoch - 1)
        metrics = " ".join(
            [f"{k}={logs.get(k):.4f}" for k in self.params["metrics"] if k in logs]
        )
        if self.enabled:
            tk.log.get(__name__).debug(
                f"Epoch {epoch + 1:3d}: lr={lr:.1e} {metrics} time={int(np.ceil(elapsed_time))} ETA={int(np.ceil(eta))}"
            )


class Checkpoint(keras.callbacks.Callback):
    """学習中に定期的に保存する。

    速度重視でinclude_optimizerはFalse固定。

    Args:
        checkpoint_path: 保存先パス
        checkpoints: 保存する回数。epochs % (checkpoints + 1) == 0だとキリのいい感じになる。

    """
    
    def __init__(self, checkpoint_path, checkpoints=3):
        super().__init__()
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoints = checkpoints
        self.target_epochs = {}
    
    def on_train_begin(self, logs=None):
        s = self.checkpoints + 1
        self.target_epochs = {self.params["epochs"] * (i + 1) // s for i in range(s)}
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.target_epochs:
            if tk.hvd.is_master():
                tk.log.get(__name__).info(
                    f"Epoch {epoch}: Saving model to {self.checkpoint_path}"
                )
                # 保存の真っ最中に死んだときに大丈夫なように少し気を使った処理をする。
                # 一度.tmpに保存してから元のを.bkにリネームして.tmpを外して.bkを削除。
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.checkpoint_path.parent / (
                    self.checkpoint_path.name + ".tmp"
                )
                backup_path = self.checkpoint_path.parent / (
                    self.checkpoint_path.name + ".bk"
                )
                if temp_path.exists():
                    temp_path.unlink()
                if backup_path.exists():
                    backup_path.unlink()
                self.model.save(str(temp_path), include_optimizer=False)
                if self.checkpoint_path.exists():
                    self.checkpoint_path.rename(backup_path)
                temp_path.rename(self.checkpoint_path)
                if backup_path.exists():
                    backup_path.unlink()
            tk.hvd.barrier()


class ErrorOnNaN(keras.callbacks.Callback):
    """NaNやinfで異常終了させる。"""
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError(f"Batch {batch}: Invalid loss")


class AUCCallback(keras.callbacks.Callback):
    """AUCを計算する
    EpochLoggerで表示してくれなくてかなC
    self.params["metrics"]に登録が必要だけど...
    
    Args:
        val_set (tk.data.Dataset):
        val_data_loader (tk.data.DataLoader):
        class_name (list):
        use_horovod (bool):
        check_epoch (int):
        use_classname (bool):

    """
    def __init__(self,
                 val_set,
                 val_data_loader, class_names, use_classname=True,use_horovod = False,check_epoch=2):
        super().__init__()
        self.val_data = val_set
        self.val_loader = val_data_loader
        self.class_names = class_names
        self.use_horovod = use_horovod
        self.mean_val = 0
        self.val_list = [0 for i in range(len(class_names))]
        self.num_classes=len(class_names)
        self.call_epochs=check_epoch
        self.use_classname=use_classname
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # if epoch in self.target_epochs
        if epoch % self.call_epochs == 0:
            y_pred_val = self.predict(self.val_data, self.val_loader)
            meanscore = sklearn.metrics.roc_auc_score(self.val_data.labels, y_pred_val)
            score_list = sklearn.metrics.roc_auc_score(self.val_data.labels, y_pred_val,average=None)
            logs["end_mauc"] = meanscore
            self.mean_val = meanscore
            self.val_list = score_list
            for i in range(self.num_classes):
                logs["end_auc_{}".format(i)] = score_list[i]

            if tk.hvd.is_master():
                meanscore = logs.get("end_mauc")
                print_str=" —end_mauc: {:f},".format(meanscore)
                if self.use_classname:
                    for i, name in enumerate(self.class_names):
                        score = logs.get("end_auc_{}".format(i))
                        if i==(self.num_classes-1):
                            print_str+=" —end_auc_{0}: {1:f}".format(name, score)
                        else:
                            print_str += " —end_auc_{0}: {1:f},".format(name, score)
                else:
                    for i in range(self.num_classes):
                        score = logs.get("end_auc_{}".format(i))
                        if i==(self.num_classes-1):
                            print_str+=" —end_auc_{0}: {1:f}".format(i, score)
                        else:
                            print_str += " —end_auc_{0}: {1:f},".format(i, score)
                tk.log.get(__name__).info(print_str)
            self.params['metrics'].append('end_mauc')
            [self.params['metrics'].append("end_auc_{}".format(i)) for i in range(self.num_classes)]
            tk.hvd.barrier()
        else:
            logs["end_mauc"] = self.mean_val
            for i in range(self.num_classes):
                logs["end_auc_{}".format(i)] = self.val_list[i]
            self.params['metrics'].append('end_mauc')
            [self.params['metrics'].append("end_auc_{}".format(i)) for i in range(self.num_classes)]
    
    def predict(self, dataset, data_loader):
        with tk.log.trace_scope("auc_predict"):
            dataset = tk.hvd.split(dataset) if self.use_horovod else dataset
            iterator = data_loader.iter(dataset)
            values = tk.models._predict_flow(
                self.model, iterator, 1 if tk.hvd.is_master() else 0, None, desc="auc_predict"
            )
            values = np.array(list(values))
            
            values = tk.hvd.allgather(values) if self.use_horovod else values
            return values
