"""Darknet53。"""

from ..dl import hvd

WEIGHTS_NAME = 'pytoolkit_darknet53_weights.h5'
WEIGHTS_URL = 'https://github.com/ak110/object_detector/releases/download/v0.0.1/darknet53_weights.h5'
WEIGHTS_HASH = '1a3857c961bcb77cd25ebbe0fcb346d4'


def darknet53(include_top=False, input_shape=None, input_tensor=None, weights='imagenet'):
    """Darknet53。"""
    import keras
    from . import yolov3

    if input_shape is not None:
        assert input_tensor is None
        input_tensor = keras.layers.Input(input_shape)
    assert not include_top  # Trueは未対応
    assert input_tensor is not None
    assert weights in (None, 'imagenet')

    x = yolov3.darknet_body(input_tensor)
    model = keras.models.Model(input_tensor, x)

    if weights == 'imagenet':
        weights_path = hvd.get_file(WEIGHTS_NAME, WEIGHTS_URL, file_hash=WEIGHTS_HASH, cache_subdir='models')
        model.load_weights(str(weights_path))

    return model


def preprocess_input(x):
    """前処理。"""
    return x / 255
