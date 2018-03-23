"""Kerasのカスタムレイヤー。"""

import numpy as np


def group_normalization():
    """Group normalization。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

    class GroupNormalization(keras.engine.topology.Layer):
        """Group normalization。

        # Arguments
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor.
                If False, `beta` is ignored.
            scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
                When the next layer is linear (also e.g. `nn.relu`),
                this can be disabled since the scaling
                will be done by the next layer.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.

        # Input shape
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        # Output shape
            Same shape as input.

        # References
            - [Group Normalization](https://arxiv.org/abs/1803.08494)
        """

        def __init__(self,
                     groups=32,
                     epsilon=1e-3,
                     center=True,
                     scale=True,
                     beta_initializer='zeros',
                     gamma_initializer='ones',
                     beta_regularizer=None,
                     gamma_regularizer=None,
                     beta_constraint=None,
                     gamma_constraint=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = keras.initializers.get(beta_initializer)
            self.gamma_initializer = keras.initializers.get(gamma_initializer)
            self.beta_regularizer = keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = keras.constraints.get(beta_constraint)
            self.gamma_constraint = keras.constraints.get(gamma_constraint)
            self.gamma = None
            self.beta = None

        def build(self, input_shape):
            dim = input_shape[-1]
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(shape=shape,
                                             name='gamma',
                                             initializer=self.gamma_initializer,
                                             regularizer=self.gamma_regularizer,
                                             constraint=self.gamma_constraint)
            else:
                self.gamma = None
            if self.center:
                self.beta = self.add_weight(shape=shape,
                                            name='beta',
                                            initializer=self.beta_initializer,
                                            regularizer=self.beta_regularizer,
                                            constraint=self.beta_constraint)
            else:
                self.beta = None
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            x = inputs
            shape = K.shape(x)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            x = K.reshape(x, [N, H, W, self.groups, C // self.groups])
            mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, H, W, C])
            return x * self.gamma + self.beta

        def get_config(self):
            config = {
                'groups': self.groups,
                'epsilon': self.epsilon,
                'center': self.center,
                'scale': self.scale,
                'beta_initializer': keras.initializers.serialize(self.beta_initializer),
                'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
                'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
                'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
                'beta_constraint': keras.constraints.serialize(self.beta_constraint),
                'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return GroupNormalization


def destandarization():
    """事前に求めた平均と標準偏差を元に出力を標準化するレイヤー。"""
    import keras
    import keras.backend as K

    class Destandarization(keras.engine.topology.Layer):
        """事前に求めた平均と標準偏差を元に出力を標準化するレイヤー。

        # Arguments
            - mean: 平均 (float).
            - std: 標準偏差 (positive float).

        # Input shape
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        # Output shape
            Same shape as input.
        """

        def __init__(self, mean=0, std=0.3, **kwargs):
            self.supports_masking = True
            self.mean = K.cast_to_floatx(mean)
            self.std = K.cast_to_floatx(std)
            if self.std <= K.epsilon():
                self.std = 1.  # 怪しい安全装置
            super().__init__(**kwargs)

        def call(self, inputs, **kwargs):
            return inputs * self.std + self.mean

        def get_config(self):
            config = {'mean': float(self.mean), 'std': float(self.std)}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return Destandarization


def stocastic_add():
    """Stocastic Depthのための確率的な加算。"""
    import keras
    import keras.backend as K

    class StocasticAdd(keras.engine.topology.Layer):
        """Stocastic Depthのための確率的な加算。

        # 引数
        - p: survival probability。1だとdropせず、0.5だと1/2の確率でdrop

        """

        def __init__(self, survival_prob, calibration=True, **kargs):
            assert 0 < survival_prob <= 1
            self.survival_prob = float(survival_prob)
            self.calibration = calibration
            super().__init__(**kargs)

        def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
            assert len(inputs) == 2

            if self.survival_prob >= 1:
                return inputs[0] + inputs[1]

            def _add():
                return inputs[0] + inputs[1]

            def _drop():
                return inputs[0]

            def _stocastic_add():
                r = K.random_uniform((1,), 0, 1)[0]
                return K.switch(K.less_equal(r, self.survival_prob), _add, _drop)

            def _calibrated_add():
                if self.calibration:
                    return inputs[0] + inputs[1] * self.survival_prob
                else:
                    return inputs[0] + inputs[1]

            return K.in_train_phase(_stocastic_add, _calibrated_add, training=training)

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            assert input_shape[0] == input_shape[1]
            return input_shape[0]

        def get_config(self):
            config = {
                'survival_prob': self.survival_prob,
                'calibration': self.calibration,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return StocasticAdd


def normal_noise():
    """平均0、分散1のノイズをドロップアウト風に適用する。"""
    import keras
    import keras.backend as K

    class NormalNoise(keras.engine.topology.Layer):
        """平均0、分散1のノイズをドロップアウト風に適用する。"""

        def __init__(self, noise_rate=0.25, **kargs):
            assert 0 <= noise_rate < 1
            self.noise_rate = noise_rate
            super().__init__(**kargs)

        def call(self, inputs, training=None):  # pylint: disable=arguments-differ
            if self.noise_rate <= 0:
                return inputs

            def _passthru():
                return inputs

            def _erase_random():
                shape = K.shape(inputs)
                noise = K.random_normal(shape)
                mask = K.cast(K.greater_equal(K.random_uniform(shape), self.noise_rate), K.floatx())
                return inputs * mask + noise * (1 - mask)

            return K.in_train_phase(_erase_random, _passthru, training=training)

        def get_config(self):
            config = {'noise_rate': self.noise_rate, }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NormalNoise


def l2normalization():
    """L2 Normalizationレイヤー。"""
    import keras
    import keras.backend as K

    class L2Normalization(keras.engine.topology.Layer):
        """L2 Normalizationレイヤー。"""

        def __init__(self, scale=1, **kargs):
            self.scale = scale
            self.gamma = None
            super().__init__(**kargs)

        def build(self, input_shape):
            ch_axis = 3 if K.image_data_format() == 'channels_last' else 1
            shape = (input_shape[ch_axis],)
            init_gamma = self.scale * np.ones(shape)
            self.gamma = self.add_weight(name='gamma',
                                         shape=shape,
                                         initializer=keras.initializers.constant(init_gamma),
                                         trainable=True)
            return super().build(input_shape)

        def call(self, inputs, **kwargs):
            ch_axis = 3 if K.image_data_format() == 'channels_last' else 1
            output = K.l2_normalize(inputs, ch_axis)
            output *= self.gamma
            return output

        def get_config(self):
            config = {'scale': self.scale}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return L2Normalization


def weighted_mean():
    """入力の加重平均を取るレイヤー。"""
    import keras
    import keras.backend as K

    class WeightedMean(keras.engine.topology.Layer):
        """入力の加重平均を取るレイヤー。"""

        def __init__(self,
                     kernel_initializer=keras.initializers.constant(0.1),
                     kernel_regularizer=None,
                     kernel_constraint='non_neg',
                     **kwargs):
            self.supports_masking = True
            self.kernel = None
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
            self.kernel_constraint = keras.constraints.get(kernel_constraint)
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(len(input_shape),),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            ot = K.zeros_like(inputs[0])
            for i, inp in enumerate(inputs):
                ot += inp * self.kernel[i]
            ot /= K.sum(self.kernel) + K.epsilon()
            return ot

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            return input_shape[0]

        def get_config(self):
            config = {
                'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
                'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return WeightedMean
