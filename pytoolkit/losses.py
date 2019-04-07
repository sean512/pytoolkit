"""Kerasの損失関数。"""

import numpy as np
import tensorflow as tf

from . import K, backend


def binary_crossentropy(y_true, y_pred, alpha=None):
    """クラス間のバランス補正ありのbinary_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    a_t = (y_true * alpha + (1 - y_true) * (1 - alpha)) * 2 if alpha is not None else 1
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
    return -K.sum(a_t * K.log(p_t), axis=list(range(1, K.ndim(y_true))))


def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """2クラス分類用focal loss (https://arxiv.org/abs/1708.02002)。

    Args:
        alpha (float or None): class 1の重み。論文では0.25。

    """
    a_t = (y_true * alpha + (1 - y_true) * (1 - alpha)) * 2 if alpha is not None else 1
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
    return -K.sum(a_t * K.pow(1 - p_t, gamma) * K.log(p_t), axis=list(range(1, K.ndim(y_true))))


def categorical_crossentropy(y_true, y_pred, alpha=None, class_weights=None):
    """クラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    assert alpha is None or class_weights is None  # 両方同時の指定はNG
    assert K.image_data_format() == 'channels_last'

    if alpha is None:
        if class_weights is None:
            cw = 1
        else:
            cw = np.reshape(class_weights, (1, 1, -1))
    else:
        num_classes = K.int_shape(y_pred)[-1]
        cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
        cw = np.reshape(cw, (1, 1, -1))

    y_pred = K.maximum(y_pred, K.epsilon())
    return -K.sum(y_true * K.log(y_pred) * cw, axis=list(range(1, K.ndim(y_true))))


def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """多クラス分類用focal loss (https://arxiv.org/abs/1708.02002)。

    Args:
        alpha (float or None): class 0以外の重み。論文では0.25。

    """
    assert K.image_data_format() == 'channels_last'
    if alpha is None:
        class_weights = 1
    else:
        nb_classes = K.int_shape(y_pred)[-1]
        class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
        class_weights = np.reshape(class_weights, (1, 1, -1))

    y_pred = K.maximum(y_pred, K.epsilon())
    return -K.sum(K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) * class_weights, axis=list(range(1, K.ndim(y_true))))  # pylint: disable=invalid-unary-operand-type


def lovasz_hinge(y_true, y_pred, activation='elu+1'):
    """Binary Lovasz hinge loss。"""
    return lovasz_hinge_with_logits(y_true, backend.logit(y_pred), activation=activation)


def symmetric_lovasz_hinge(y_true, y_pred, activation='elu+1'):
    """Binary Lovasz hinge lossの0, 1対称版。"""
    y_pred_logit = backend.logit(y_pred)
    loss1 = lovasz_hinge_with_logits(y_true, y_pred_logit, activation=activation)
    loss2 = lovasz_hinge_with_logits(1 - y_true, -y_pred_logit, activation=activation)
    return (loss1 + loss2) / 2


def lovasz_hinge_with_logits(y_true, y_pred_logit, activation='elu+1'):
    """Binary Lovasz hinge loss。"""
    def loss_per_image(elems):
        label, logit = elems
        logit = K.reshape(logit, (-1,))
        label = K.reshape(label, (-1,))
        label = K.cast(label, 'float32')
        signs = label * 2.0 - 1.0  # -1 ～ +1
        errors = 1.0 - logit * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0], name='sort')
        gt_sorted = K.gather(label, perm)
        gts = K.sum(gt_sorted)
        inter = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1. - gt_sorted)
        iou = 1.0 - inter / union
        grad = tf.concat((iou[0:1], iou[1:] - iou[:-1]), 0)
        if activation == 'relu':
            a = tf.nn.relu(errors_sorted)
        elif activation == 'elu+1':
            a = tf.nn.elu(errors_sorted) + 1
        else:
            raise ValueError(f'Invalid activation: {activation}')
        loss = tf.tensordot(a, tf.stop_gradient(grad), 1, name='loss_per_image')
        return loss
    return tf.map_fn(loss_per_image, (y_true, y_pred_logit), dtype=tf.float32)


def lovasz_softmax(y_true, y_pred):
    """Lovasz softmax loss。"""
    num_classes = K.int_shape(y_true)[-1]

    def loss_per_image(elems):
        label, proba = elems
        proba = K.reshape(proba, (-1, num_classes))
        label = K.reshape(label, (-1, num_classes))
        label = K.cast(label, 'float32')
        losses = []
        for c in range(num_classes):
            errors = K.abs(label[:, c] - proba[:, c])
            errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0], name=f'sort_{c}')
            gt_sorted = K.gather(label[:, c], perm)
            gts = K.sum(gt_sorted)
            inter = gts - tf.cumsum(gt_sorted)
            union = gts + tf.cumsum(1. - gt_sorted)
            iou = 1.0 - inter / union
            grad = tf.concat((iou[0:1], iou[1:] - iou[:-1]), 0)
            loss = tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name='loss_per_image')
            losses.append(loss)
        return K.mean(tf.stack(losses))
    return tf.map_fn(loss_per_image, (y_true, y_pred), dtype=tf.float32)


def mixed_lovasz_softmax(y_true, y_pred):
    """Lovasz softmax loss + CE。"""
    loss1 = lovasz_softmax(y_true, y_pred)
    loss2 = K.categorical_crossentropy(y_true, y_pred)
    return loss1 * 0.9 + loss2 * 0.1


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * K.square(y_true - y_pred)
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = K.sum(l1_loss, axis=-1)
    return l1_loss


def mse(y_true, y_pred):
    """AutoEncoderとか用mean squared error"""
    return K.mean(K.square(y_pred - y_true), axis=list(range(1, K.ndim(y_true))))


def mae(y_true, y_pred):
    """AutoEncoderとか用mean absolute error"""
    return K.mean(K.abs(y_pred - y_true), axis=list(range(1, K.ndim(y_true))))


def rmse(y_true, y_pred):
    """AutoEncoderとか用root mean squared error"""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=list(range(1, K.ndim(y_true)))))
