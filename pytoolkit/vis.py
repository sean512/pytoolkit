"""可視化関連。"""

import cv2
import numpy as np

from . import K, keras


class GradCamVisualizer:
    """Grad-CAM(のようなもの)による可視化。

    Args:
        model: 対象のモデル。画像分類で最後がpooling_class+Dense+softmaxで分類している前提。
        output_index: 使う出力(softmax)のインデックス。(クラスのindex)
        pooling_class: 分類直前のpoolingのクラス。(既定値はGlobalAveragePooling2D)
        multi_output: マルチ出力の場合にcamを作りたい出力の番号

    """

    def __init__(
        self, model: keras.models.Model, output_index: int, pooling_class: type = None,
        multi_output:int =0
    ):
        pooling_class = pooling_class or keras.layers.GlobalAveragePooling2D
        # pooling_classへの入力テンソルを取得
        map_output = None
        for layer in model.layers[::-1]:
            if isinstance(layer, pooling_class):
                map_output = layer.input
                break
        assert map_output is not None
        # 関数を作成
        if multi_output==0:
            grad = K.gradients(model.output[0, output_index], map_output)[0]
        else:
            grad = K.gradients(model.output[multi_output][0, output_index], map_output)[0]
        # mask = K.maximum(0.0, K.sum(map_output * grad, axis=-1)[0, :, :])
        # mask = mask / (K.max(mask) + 1e-3)  # [0, 1)
        mask = K.sum(map_output * grad, axis=-1)
        mask = K.relu(mask) / (K.max(mask) + K.epsilon())
        self.get_mask_func = K.function(model.inputs + [K.learning_phase()], [mask])

    def draw(
        self,
        source_image: np.ndarray,
        model_inputs: np.ndarray,
        alpha: float = 0.5,
        interpolation: str = "nearest",
    ) -> np.ndarray:
        """ヒートマップ画像を作成して返す。

        Args:
            source_image: 元画像 (RGB。shape=(height, width, 3))
            model_inputs: モデルの入力1件分。(例えば普通の画像分類ならshape=(1, height, width, 3))
            alpha: ヒートマップの不透明度
            interpolation: マスクの拡大方法 (nearest, bilinear, bicubic, lanczos)

        Returns:
            画像 (RGB。shape=(height, width, 3))

        """
        assert source_image.shape[2:] == (3,)
        assert 0 < alpha < 1
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[interpolation]

        mask = self.get_mask(model_inputs)
        mask = cv2.resize(
            mask,
            (source_image.shape[1], source_image.shape[0]),
            interpolation=cv2_interp,
        )
        mask = np.uint8(256 * mask)  # [0-255]

        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = heatmap[..., ::-1]  # BGR to RGB

        result_image = np.uint8(heatmap * alpha + source_image * (1 - alpha))
        return result_image

    def get_mask(self, model_inputs: np.ndarray) -> np.ndarray:
        """可視化してマスクを返す。マスクの値は`[0, 1)`。"""
        if not isinstance(model_inputs, list):
            model_inputs = [model_inputs]
        mask = self.get_mask_func(model_inputs + [0])[0]
        if len(mask.shape)==3:
            mask=np.squeeze(mask)
        assert len(mask.shape) == 2
        return mask
