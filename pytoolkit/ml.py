"""機械学習(主にsklearn)関連。"""
import collections
import itertools
import json
import multiprocessing as mp
import pathlib

import numpy as np
import sklearn.base
import sklearn.cluster
import sklearn.externals.joblib as joblib
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils

from . import draw, log, ndimage


class ObjectsAnnotation(object):
    """物体検出のアノテーションデータを持つためのクラス。"""

    def __init__(self, path, width, height, classes, bboxes, difficults=None):
        assert len(classes) == len(bboxes)
        assert difficults is None or len(classes) == len(difficults)
        # 画像ファイルのフルパス
        self.path = pathlib.Path(path)
        # 画像の横幅(px)
        self.width = width
        # 画像の縦幅(px)
        self.height = height
        # クラスIDの配列
        self.classes = np.asarray(classes)
        # bounding box(x1, y1, x2, y2)の配列。(0～1)
        self.bboxes = np.asarray(bboxes)
        # difficultフラグの配列。(True or False)
        self.difficults = np.asarray(difficults) if difficults is not None else np.zeros(len(classes))
        assert self.width >= 1
        assert self.height >= 1
        assert (self.bboxes >= 0).all()
        assert (self.bboxes <= 1).all()
        assert (self.bboxes[:, :2] < self.bboxes[:, 2:]).all()
        assert self.classes.shape == (self.num_objects,)
        assert self.bboxes.shape == (self.num_objects, 4)
        assert self.difficults.shape == (self.num_objects,)

    @property
    def num_objects(self):
        """物体の数を返す。"""
        return len(self.classes)

    @property
    def bboxes_ar_fixed(self):
        """縦横比を補正したbboxesを返す。(prior box算出など用)"""
        assert self.width > 0 and self.height > 0
        bboxes = np.copy(self.bboxes)
        if self.width > self.height:
            # 横長の場合、上下にパディングする想定で補正
            bboxes[:, [1, 3]] *= self.height / self.width
        elif self.width < self.height:
            # 縦長の場合、左右にパディングする想定で補正
            bboxes[:, [0, 2]] *= self.width / self.height
        return bboxes


class ObjectsPrediction(object):
    """物体検出の予測結果を持つクラス。"""

    def __init__(self, classes, confs, bboxes):
        self.classes = np.asarray(classes)
        self.confs = np.asarray(confs)
        self.bboxes = np.asarray(bboxes)
        assert self.classes.shape == (self.num_objects,)
        assert self.confs.shape == (self.num_objects,)
        assert self.bboxes.shape == (self.num_objects, 4)

    @property
    def num_objects(self):
        """物体の数を返す。"""
        return len(self.classes)

    def plot(self, img, class_names):
        """ワクを描画した画像を作って返す。"""
        return plot_objects(img, self.classes, self.confs, self.bboxes, class_names)


def listup_classification(dirpath, class_names=None):
    """画像分類でよくある、クラス名ディレクトリの列挙。クラス名の配列, X, yを返す。"""
    dirpath = pathlib.Path(dirpath)

    # クラス名
    if class_names is None:
        def _empty(it):
            for _ in it:
                return False
            return True

        class_names = list(sorted([p.name for p in dirpath.iterdir() if p.is_dir() and not _empty(p.iterdir())]))

    # 各クラスのデータ
    X, y = [], []
    for class_id, class_name in enumerate(class_names):
        t = [p for p in (dirpath / class_name).iterdir() if p.is_file()]
        X.extend(t)
        y.extend([class_id] * len(t))
    assert len(X) == len(y)
    return class_names, np.array(X), np.array(y)


def split(X, y, validation_split=None, cv_count=None, cv_index=None, split_seed=None, stratify=None):
    """データの分割。

    # 引数
    - validation_split: 実数を指定するとX, y, weightsの一部をランダムにvalidation dataとする
    - cv_count: cross validationする場合の分割数
    - cv_index: cross validationする場合の何番目か
    - split_seed: validation_splitやcvする場合のseed
    """
    assert len(X) == len(y)
    if validation_split is not None:
        # split
        assert cv_count is None
        assert cv_index is None
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X, y, test_size=validation_split, shuffle=True, random_state=split_seed, stratify=stratify)
    else:
        # cross validation
        assert cv_count is not None
        assert cv_index in range(cv_count)
        if stratify is None:
            stratify = isinstance(y, np.ndarray) and len(y.shape) == 1
        cv = sklearn.model_selection.StratifiedKFold if stratify else sklearn.model_selection.KFold
        cv = cv(cv_count, shuffle=True, random_state=split_seed)
        train_indices, val_indices = list(cv.split(X, y))[cv_index]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
    return (X_train, y_train), (X_val, y_val)


class _ToCategorical(object):
    """クラスラベルのone-hot encoding化を行うクラス。"""

    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def __call__(self, y):
        assert len(y.shape) == 1
        cat = np.zeros((len(y), self.nb_classes), dtype=np.float32)
        cat[np.arange(len(y)), y] = 1
        return cat


def to_categorical(nb_classes):
    """クラスラベルのone-hot encoding化を行う関数を返す。"""
    return _ToCategorical(nb_classes)


def compute_map(gt, pred, iou_threshold=0.5, use_voc2007_metric=False):
    """`mAP`の算出。

    # 引数
    - gt_classes_list: 正解のbounding box毎のクラス。shape=(画像数, bbox数)
    - gt_bboxes_list: 正解のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)
    - gt_difficults_list: 正解のbounding boxにdifficultフラグがついているか否か。shape=(画像数, bbox数)
    - pred_classes_list: 予測結果のbounding box毎のクラス。shape=(画像数, bbox数)
    - pred_confs_list: 予測結果のbounding box毎のconfidence。shape=(画像数, bbox数)
    - pred_bboxes_list: 予測結果のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)

    """
    assert len(gt) == len(pred)
    gt_classes_list = np.array([y.classes for y in gt])
    gt_bboxes_list = np.array([y.bboxes for y in gt])
    gt_difficults_list = np.array([y.difficults for y in gt])
    pred_classes_list = np.array([p.classes for p in pred])
    pred_confs_list = np.array([p.confs for p in pred])
    pred_bboxes_list = np.array([p.bboxes for p in pred])

    npos_dict = collections.defaultdict(int)
    scores = collections.defaultdict(list)
    matches = collections.defaultdict(list)

    for gt_classes, gt_bboxes, gt_difficults, pred_classes, pred_confs, pred_bboxes in zip(
            gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_confs_list, pred_bboxes_list):

        for class_id in np.unique(np.concatenate([gt_classes, pred_classes])):
            pred_mask = pred_classes == class_id
            pred_confs_class = pred_confs[pred_mask]
            pred_bboxes_class = pred_bboxes[pred_mask]
            order = pred_confs_class.argsort()[::-1]
            pred_confs_class = pred_confs_class[order]
            pred_bboxes_class = pred_bboxes_class[order]

            gt_mask = gt_classes == class_id
            gt_bboxes_class = gt_bboxes[gt_mask]
            gt_difficults_class = gt_difficults[gt_mask]

            npos_dict[class_id] += np.logical_not(gt_difficults_class).sum()
            scores[class_id].extend(pred_confs_class)

            if len(pred_bboxes_class) == 0:
                continue
            if len(gt_bboxes_class) == 0:
                matches[class_id].extend([0] * len(pred_bboxes_class))
                continue

            iou = compute_iou(pred_bboxes_class, gt_bboxes_class)
            gt_indices = iou.argmax(axis=1)
            gt_indices[iou.max(axis=1) < iou_threshold] = -1  # 不一致

            detected = np.zeros(len(gt_bboxes_class), dtype=bool)
            for gt_i in gt_indices:
                if gt_i >= 0:
                    if gt_difficults_class[gt_i]:
                        matches[class_id].append(-1)  # skip
                    else:
                        matches[class_id].append(0 if detected[gt_i] else 1)
                        detected[gt_i] = True  # 2回目以降は不一致扱い
                else:
                    matches[class_id].append(0)  # 不一致

    precision = []
    recall = []
    for class_id, npos in npos_dict.items():
        order = np.array(scores[class_id]).argsort()[::-1]
        matches_class = np.array(matches[class_id])[order]

        tp = np.cumsum(matches_class == 1)
        fp = np.cumsum(matches_class == 0)
        tpfp = fp + tp
        tpfp = np.where(tpfp == 0, [np.nan] * len(tpfp), tpfp)  # 警告対策 (0ならnanにする)

        precision.append(tp / tpfp)
        recall.append(None if npos == 0 else tp / npos)

    ap = [compute_ap(prec, rec, use_voc2007_metric) for prec, rec in zip(precision, recall)]
    return np.nanmean(ap)


def compute_ap(precision, recall, use_voc2007_metric=False):
    """Average precisionの算出。"""
    if precision is None or recall is None:
        return np.nan
    if use_voc2007_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(precision)[recall >= t])
            ap += p
        ap /= 11
    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # pylint: disable=no-member

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_scores(gt, pred, iou_threshold=0.5):
    """物体検出の正解と予測結果から、適合率、再現率、F値、該当回数を算出して返す。"""
    assert len(gt) == len(pred)
    num_classes = len(np.unique(np.concatenate([y.classes for y in gt], axis=0)))

    tp = np.zeros((num_classes,), dtype=int)  # true positive
    fp = np.zeros((num_classes,), dtype=int)  # false positive
    tn = np.zeros((num_classes,), dtype=int)  # true negative

    for y_true, y_pred in zip(gt, pred):
        pred_mask = np.ones((y_pred.num_objects,), dtype=bool)
        # 各正解が予測結果に含まれるか否か: true positive/negative
        for gt_class, gt_bbox, gt_difficult in zip(y_true.classes, y_true.bboxes, y_true.difficults):
            pred_bboxes = y_pred.bboxes[np.logical_and(pred_mask, y_pred.classes == gt_class)]
            if not pred_bboxes.any():
                continue
            iou = compute_iou(np.expand_dims(gt_bbox, axis=0), pred_bboxes)[0, :]
            pred_ix = iou.argmax()
            if iou[pred_ix] >= iou_threshold:
                # 検出成功
                if not gt_difficult:
                    tp[gt_class] += 1
                pred_mask[pred_ix] = False
            else:
                # 検出失敗
                if not gt_difficult:
                    tn[gt_class] += 1
        # 正解に含まれなかった予測結果: false positive
        for pred_class in y_pred.classes[pred_mask]:
            fp[pred_class] += 1

    supports = tp + tn
    precisions = tp.astype(float) / (tp + fp)
    recalls = tp.astype(float) / supports
    fscores = 2 / (1 / (precisions + 1e-7) + 1 / (recalls + 1e-7))
    return precisions, recalls, fscores, supports


def print_scores(precisions, recalls, fscores, supports, class_names=None, print_fn=None):
    """適合率・再現率などをprintする。(classification_report風。)"""
    assert len(precisions) == len(recalls)
    assert len(precisions) == len(fscores)
    assert len(precisions) == len(supports)
    class_names = class_names or [f'class{i:02d}' for i in range(len(precisions))]
    print_fn = print_fn or log.get(__name__).info

    print_fn('                   適合率  再現率  F値    件数')
    # .......'0123456789abcdef:  0.123   0.123   0.123  0123456'

    for cn, prec, rec, f1, sup in zip(class_names, precisions, recalls, fscores, supports):
        print_fn(f'{cn:16s}:  {prec:.3f}   {rec:.3f}   {f1:.3f}  {sup:7d}')

    cn = 'avg / total'
    prec = np.average(precisions, weights=supports)
    rec = np.average(recalls, weights=supports)
    f1 = 2 / (1 / (prec + 1e-7) + 1 / (rec + 1e-7))
    sup = np.sum(supports)
    print_fn(f'{cn:16s}:  {prec:.3f}   {rec:.3f}   {f1:.3f}  {sup:7d}')


def top_k_accuracy(y_true, proba_pred, k=5):
    """Top-K accuracy。"""
    assert len(y_true.shape) == 1
    assert len(proba_pred.shape) == 2
    best_k = np.argsort(proba_pred, axis=1)[:, -k:]
    return np.mean([y in best_k[i, :] for i, y in enumerate(y_true)])


def bboxes_center(bboxes):
    """Bounding boxの中心を返す。"""
    assert bboxes.shape[-1] == 4
    return (bboxes[..., 2:] + bboxes[..., :2]) / 2


def bboxes_size(bboxes):
    """Bounding boxのサイズを返す。"""
    assert bboxes.shape[-1] == 4
    return bboxes[..., 2:] - bboxes[..., :2]


def bboxes_area(bboxes):
    """Bounding boxの面積を返す。"""
    return bboxes_size(bboxes).prod(axis=-1)


def compute_iou(bboxes_a, bboxes_b):
    """IoU(Intersection over union、Jaccard係数)の算出。

    重なり具合を示す係数。(0～1)
    """
    assert bboxes_a.shape[0] > 0
    assert bboxes_b.shape[0] > 0
    assert bboxes_a.shape == (len(bboxes_a), 4)
    assert bboxes_b.shape == (len(bboxes_b), 4)
    lt = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    rb = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    area_inter = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    area_union = area_a[:, np.newaxis] + area_b - area_inter
    iou = area_inter / area_union
    assert iou.shape == (len(bboxes_a), len(bboxes_b))
    return iou


def compute_size_based_iou(bbox_sizes_a, bbox_sizes_b):
    """中心が一致している場合のIoUを算出する。

    例えば(10, 15)と(15, 10)で0.5。
    """
    assert bbox_sizes_a.shape[0] > 0
    assert bbox_sizes_b.shape[0] > 0
    if bbox_sizes_a.shape[-1] == 4 and bbox_sizes_b.shape[-1] == 4:
        bbox_sizes_a = bbox_sizes_a[:, 2:] - bbox_sizes_a[:, :2]
        bbox_sizes_b = bbox_sizes_b[:, 2:] - bbox_sizes_b[:, :2]
    assert bbox_sizes_a.shape == (len(bbox_sizes_a), 2)
    assert bbox_sizes_b.shape == (len(bbox_sizes_b), 2)
    area_inter = np.prod(np.minimum(bbox_sizes_a[:, np.newaxis, :], bbox_sizes_b), axis=-1)
    area_a = np.prod(bbox_sizes_a, axis=-1)
    area_b = np.prod(bbox_sizes_b, axis=-1)
    area_union = area_a[:, np.newaxis] + area_b - area_inter
    iou = area_inter / area_union
    assert iou.shape == (len(bbox_sizes_a), len(bbox_sizes_b))
    return iou


def is_intersection(bboxes_a, bboxes_b):
    """boxes_aとboxes_bでそれぞれ交差している部分が存在するか否かを返す。"""
    assert bboxes_a.shape[0] > 0
    assert bboxes_b.shape[0] > 0
    assert bboxes_a.shape == (len(bboxes_a), 4)
    assert bboxes_b.shape == (len(bboxes_b), 4)
    lt = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    rb = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    return (lt < rb).all(axis=-1)


def is_in_box(boxes_a, boxes_b):
    """boxes_aがboxes_bの中に完全に入っているならtrue。"""
    assert boxes_a.shape == (len(boxes_a), 4)
    assert boxes_b.shape == (len(boxes_b), 4)
    lt = boxes_a[:, np.newaxis, :2] >= boxes_b[:, :2]
    rb = boxes_a[:, np.newaxis, 2:] <= boxes_b[:, 2:]
    return np.logical_and(lt, rb).all(axis=-1)


def cluster_by_iou(X, n_clusters, **kwargs):
    """`1 - IoU`の値によるクラスタリング。

    YOLOv2のDimension Clustersで使用されているもの。
    KMeansのモデルのインスタンスを返す。
    """
    def _iou_distances(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
        """`1 - IoU`を返す。"""
        if Y is None:
            return _iou_distances(X, X, Y_norm_squared, squared, X_norm_squared)
        if len(X.shape) == 1:
            return _iou_distances(np.expand_dims(X, 0), Y, Y_norm_squared, squared, X_norm_squared)
        assert X.shape == (len(X), 2)
        assert Y.shape == (len(Y), 2)
        dist = 1 - compute_size_based_iou(X, Y)
        return np.square(dist) if squared else dist

    from sklearn.cluster import k_means_
    old = k_means_.euclidean_distances
    k_means_.euclidean_distances = _iou_distances  # monkey-patch
    model = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs)
    model.fit(X)
    k_means_.euclidean_distances = old
    return model


def non_maximum_suppression(boxes, scores, top_k=200, iou_threshold=0.45):
    """`iou_threshold`分以上重なっている場合、スコアの大きい方のみ採用する。

    # 引数
    - boxes: ボックスの座標。shape=(ボックス数, 4)
    - scores: スコア。shape=(ボックス数,)
    - top_k: 最大何個の結果を返すか。
    - iou_threshold: 重なりの閾値。

    # 戻り値
    インデックス。

    """
    assert len(boxes) == len(scores)
    assert boxes.shape == (len(scores), 4)
    sorted_indices = scores.argsort()[::-1]
    boxes = boxes[sorted_indices]
    boxes_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

    selected = []
    for i, b in enumerate(boxes):
        if (b[2:] - b[:2] <= 0).any():
            continue  # 面積のないboxはskip
        if len(selected) == 0:
            selected.append(i)
        else:
            lt = np.maximum(b[:2], boxes[selected, :2])
            rb = np.minimum(b[2:], boxes[selected, 2:])
            area_inter = np.prod(rb - lt, axis=1) * (lt < rb).all(axis=1)
            iou = area_inter / (boxes_area[i] + boxes_area[selected] - area_inter)
            if (iou >= iou_threshold).any():
                continue  # 重なっているのでskip
            selected.append(i)
            if len(selected) >= top_k:
                break

    return sorted_indices[selected]


class WeakModel(object):
    """CVしたりout-of-folds predictionを作ったりするクラス。"""

    def __init__(self, model_dir, base_estimator, cv=5, fit_params=None):
        self.model_dir = pathlib.Path(model_dir)
        self.base_estimator = base_estimator
        self.cv = cv
        self.fit_params = fit_params
        self.estimators_ = None
        self.data_ = {}

    def fit(self, X, y, groups=None, pool=None):
        """学習"""
        if not pool:
            pool = mp.Pool()
        func, args = self.make_fit_tasks(X, y, groups)
        pool.map(func, args)

    def make_fit_tasks(self, X, y, groups=None):
        """学習の処理を作って返す。(func, args)形式。"""
        self._init_data()
        args = []
        for fold in range(self.cv):
            estimator = sklearn.base.clone(self.base_estimator)
            args.append((estimator, fold, X, y, groups))
        return self._fit, args

    def split(self, fold, X, y, groups=None):
        """データの分割"""
        rs = np.random.RandomState(self.data_['split_seed'])
        classifier = sklearn.base.is_classifier(self.base_estimator)
        # cv = sklearn.model_selection.check_cv(self.cv, y, classifier=classifier)
        if classifier:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=rs)
        else:
            cv = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=rs)
        return list(cv.split(X, y, groups))[fold]

    def _fit(self, estimator, fold, X, y, groups=None):
        """学習。"""
        X, y, groups = sklearn.utils.indexable(X, y, groups)  # pylint: disable=E0632
        fit_params = self.fit_params if self.fit_params is not None else {}

        train, _ = self.split(fold, X, y, groups)
        estimator.fit(X[train], y[train], **fit_params)
        pred = estimator.predict_proba(X)
        joblib.dump(pred, str(self.model_dir / f'predict.fold{fold}.train.pkl'))
        joblib.dump(estimator, str(self.model_dir / f'model.fold{fold}.pkl'))

    def oopf(self, X, y, groups=None):
        """out-of-folds predictionを作って返す。

        Xはデータの順序が変わってないかのチェック用。
        """
        self._init_data()
        oopf = joblib.load(str(self.model_dir / f'predict.fold{0}.train.pkl'))
        for fold in range(1, self.cv):
            pred = joblib.load(str(self.model_dir / f'predict.fold{fold}.train.pkl'))
            _, test = self.split(fold, X, y, groups)
            oopf[test] = pred[test]
        return oopf

    def _init_data(self):
        """self.data_の初期化。今のところ(?)split_seedのみ。"""
        model_json_file = self.model_dir / 'model.json'
        if model_json_file.is_file():
            # あれば読み込み
            with model_json_file.open() as f:
                self.data_ = json.load(f)
        else:
            # 無ければ生成して保存
            self.data_ = {
                'split_seed': int(np.random.randint(0, 2 ** 31)),
            }
            with model_json_file.open('w') as f:
                json.dump(self.data_, f, indent=4, sort_keys=True)


def print_classification_metrics(y_true, proba_pred, print_fn=None):
    """分類の指標色々を表示する。"""
    print_fn = print_fn or log.get(__name__).info
    pred_type = sklearn.utils.multiclass.type_of_target(proba_pred)
    assert pred_type in ('continuous', 'continuous-multioutput')
    if pred_type == 'continuous':  # binary
        y_pred = (np.asarray(proba_pred) >= 0.5).astype(np.int32)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        auc = sklearn.metrics.roc_auc_score(y_true, proba_pred)
        logloss = sklearn.metrics.log_loss(y_true, proba_pred)
        print_fn(f'Accuracy: {acc:.3f}')
        print_fn(f'F1-score: {f1:.3f}')
        print_fn(f'AUC:      {auc:.3f}')
        print_fn(f'Logloss:  {logloss:.3f}')
    else:  # multiclass
        y_pred = np.argmax(proba_pred, axis=-1)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1mi = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
        f1ma = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        logloss = sklearn.metrics.log_loss(y_true, proba_pred)
        print_fn(f'Accuracy: {acc:.3f}')
        print_fn(f'F1-micro: {f1mi:.3f}')  # クラス間の重み付き平均
        print_fn(f'F1-macro: {f1ma:.3f}')  # クラス間の平均(balanced accuracyみたいなやつ)
        print_fn(f'Logloss:  {logloss:.3f}')


def plot_cm(cm, to_file='confusion_matrix.png', classes=None, normalize=True, title='Confusion matrix'):
    """Confusion matrixを画像化する。

    参考: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if classes is None:
        classes = [f'class {i}' for i in range(len(cm))]

    if normalize:
        cm = np.array(cm, dtype=np.float32)
        cm /= cm.sum(axis=1)[:, np.newaxis]  # normalize
    else:
        cm = np.array(cm)

    size = (len(cm) + 1) // 2
    fig = draw.create_figure(figsize=(size + 4, size + 2), dpi=96)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(res)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)

    to_file = pathlib.Path(to_file)
    to_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(to_file), bbox_inches='tight')
    fig.clf()


def plot_objects(base_image, classes, confs, bboxes, class_names):
    """画像＋オブジェクト([class_id + confidence + xmin/ymin/xmax/ymax]×n)を画像化する。

    # 引数
    - base_image: 元画像ファイルのパスまたはndarray
    - classes: クラスIDのリスト
    - confs: confidenceのリスト (None可)
    - bboxes: xmin/ymin/xmax/ymaxのリスト (それぞれ0.0 ～ 1.0)
    - class_names: クラスID→クラス名のリスト  (None可)

    """
    import cv2

    if confs is None:
        confs = [None] * len(classes)
    assert len(classes) == len(confs)
    assert len(classes) == len(bboxes)
    if class_names is not None and any(classes):
        assert 0 <= np.min(classes) < len(class_names)
        assert 0 <= np.max(classes) < len(class_names)

    img = ndimage.load(base_image, grayscale=False)
    colors = draw.get_colors(len(class_names) if class_names is not None else 1)

    for classid, conf, bbox in zip(classes, confs, bboxes):
        xmin = int(round(bbox[0] * img.shape[1]))
        ymin = int(round(bbox[1] * img.shape[0]))
        xmax = int(round(bbox[2] * img.shape[1]))
        ymax = int(round(bbox[3] * img.shape[0]))
        label = class_names[classid] if class_names is not None else f'class{classid:02d}'
        color = colors[classid % len(colors)][-2::-1]  # RGBA → BGR
        text = label if conf is None else f'{conf:0.2f}, {label}'

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=2)

        tw = 6 * len(text)
        cv2.rectangle(img, (xmin - 1, ymin - 10), (xmin + tw + 15, ymin + 5), color, -1)
        cv2.putText(img, text, (xmin + 5, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return img
