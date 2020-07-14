from typing import Tuple, List
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Input,
                                            ZeroPadding2D,
                                            Conv2D,
                                            MaxPool2D,
                                            Add,
                                            BatchNormalization,
                                            LeakyReLU,
                                            UpSampling2D,
                                            Concatenate,
                                            Reshape)
from tensorflow.python.keras.regularizers import l2


class WeightLoader(object):
    _restore = False

    def __init__(self):
        self.cnt = 0
        self.weights = None

    def load_weights(self, path: str) -> None:
        with open(path, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
            self.weights = np.fromfile(wf, dtype=np.float32)

    def __call__(self, num_weights: int) -> np.ndarray:
        weights = self.weights[self.cnt:self.cnt + num_weights]
        self.cnt += int(num_weights)
        return weights

    def reset(self) -> None:
        self.cnt = 0


_weight_loader = WeightLoader()


def DarknetConv2D(filters: int = 128,
                  kernel_size: Tuple[int, int] = (3, 3),
                  strides: Tuple[int, int] = (1, 1),
                  pad: str = 'same',
                  use_bn: bool = True,
                  skip_weights: bool = False,
                  trainable: bool = True,
                  dtype: str = tf.keras.backend.floatx()) -> callable:
    _filters = filters

    def build(input_shape: Tuple[int, int, int, int], filters: int) -> List[callable]:

        conv_weights = None
        bn_weights = None
        ops = []
        if _weight_loader._restore:

            if skip_weights:
                filters = 3 * (5 + 80)

            conv_shape = (filters, input_shape[-1], kernel_size[0], kernel_size[1])

            if use_bn:
                bn_weights = _weight_loader(4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                conv_weights = _weight_loader(int(np.prod(conv_shape)))
                conv_weights = [conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])]

            else:
                conv_bias = _weight_loader(filters)
                conv_weights = _weight_loader(int(np.prod(conv_shape)))
                conv_weights = [conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0]), conv_bias]

            if skip_weights:
                conv_weights = None
                bn_weights = None

        ops.append(Conv2D(_filters, kernel_size,
                           padding=pad,
                           strides=strides,
                           activation='linear',
                           kernel_regularizer=l2(1e-5),
                           use_bias=not use_bn,
                           weights=conv_weights,
                           trainable=trainable,
                           dtype=dtype))
        if use_bn:
            ops.append(BatchNormalization(weights=bn_weights, trainable=trainable, dtype=dtype))
            ops.append(LeakyReLU(0.1))
        return ops

    def call(inputs: tf.Tensor, **kwargs) -> tf.Tensor:

        ops = build(inputs.get_shape(), filters)

        x = inputs
        for op in ops:
            x = op(x, **kwargs)
        return x

    return call


def DarknetResidual(filters: int, trainable: bool = True) -> callable:

    def call(inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = DarknetConv2D(filters // 2, (1, 1), trainable=trainable)(inputs, **kwargs)
        x = DarknetConv2D(filters, (3, 3), trainable=trainable)(x, **kwargs)
        x = Add()([inputs, x])
        return x

    return call


def SPPLayer() -> callable:

    def call(inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x_5 = MaxPool2D(pool_size=(5, 5), strides=1, padding='same')(inputs)
        x_9 = MaxPool2D(pool_size=(9, 9), strides=1, padding='same')(inputs)
        x_13 = MaxPool2D(pool_size=(13, 13), strides=1, padding='same')(inputs)
        x = Concatenate()([x_13, x_9, x_5, inputs])
        return x
    return call


def YOLOBlock(filters, output_filters, name, use_spp=False, trainable=True, restore_output_weights=False) -> callable:

    def build(input_shape: Tuple[int, int, int, int]) -> Tuple[List[callable], List[callable]]:
        ops = []
        output_ops = []
        ops.append(DarknetConv2D(filters, (1, 1), trainable=trainable))
        ops.append(DarknetConv2D(filters * 2, (3, 3), trainable=trainable))
        ops.append(DarknetConv2D(filters, (1, 1), trainable=trainable))
        if use_spp:
            ops.append(SPPLayer())
            ops.append(DarknetConv2D(filters, (1, 1)))
        ops.append(DarknetConv2D(filters * 2, (3, 3), trainable=trainable))
        ops.append(DarknetConv2D(filters, (1, 1), trainable=trainable))
        output_ops.append(DarknetConv2D(filters * 2, (3, 3), trainable=trainable))
        output_ops.append(
            DarknetConv2D(output_filters, (1, 1), use_bn=False, skip_weights=not restore_output_weights, trainable=trainable, dtype=tf.float32))
        output_ops.append(Reshape([*input_shape[1:-1], 3, output_filters // 3], name=name, dtype=tf.float32))

        return ops, output_ops

    def call(inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        ops, output_ops = build(inputs.get_shape())
        x = inputs
        for op in ops:
            x = op(x, **kwargs)

        fmap = x
        for fmap_op in output_ops:
            fmap = fmap_op(fmap, **kwargs)

        return x, fmap
    return call


def DarknetBlock(filters: int, n_blocks: int, trainable: bool = True) -> callable:

    def build(input_shape: Tuple[int, int, int, int]) -> List[callable]:
        ops = []
        ops.append(ZeroPadding2D(((1, 0), (1, 0))))
        ops.append(DarknetConv2D(filters, (3, 3), strides=(2, 2), pad='valid', trainable=trainable))
        for i in range(n_blocks):
            ops.append(DarknetResidual(filters, trainable=trainable))
        return ops

    def call(inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = inputs
        ops = build(inputs.get_shape())
        for op in ops:
            x = op(x, **kwargs)

        return x
    return call


def Darknet53Body(trainable: bool) -> callable:

    filters_per_level = [64, 128, 256, 512, 1024]
    n_blocks_per_level = [1, 2, 8, 8, 4]

    def call(inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        x = DarknetConv2D(32, (3, 3), trainable=trainable)(inputs)

        fmaps = []
        for filters, n_blocks in zip(filters_per_level, n_blocks_per_level):
            x = DarknetBlock(filters, n_blocks, trainable=trainable)(x, **kwargs)
            fmaps.append(x)

        x_52, x_26, x_13 = fmaps[-3:]

        return x_52, x_26, x_13
    return call


def Upsample(filters: int, trainable: bool = True) -> callable:

    def call(inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        x, lateral_x = inputs

        x = DarknetConv2D(filters, (1, 1), trainable=trainable)(x, **kwargs)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, lateral_x])
        return x
    return call


def YOLOv3Net(restore_weights: bool = True,
              trainable_backbone: bool = True,
              n_cls: int = 80,
              use_spp: bool = True,
              restore_output_weights: bool = True) -> callable:

    if restore_weights:
        path = 'yolov3-spp.weights' if use_spp else 'yolov3.weights'
        _weight_loader.load_weights(path)
        _weight_loader._restore = True
        print('Restoring weights...')

    def call(inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        x_52, x_26, x_13 = Darknet53Body(trainable=trainable_backbone)(inputs, **kwargs)

        x, fmap_13 = YOLOBlock(512, 3 * (5 + n_cls), 'y_13', trainable=True,
                               use_spp=use_spp, restore_output_weights=restore_output_weights)(x_13, **kwargs)
        x = Upsample(256, trainable=True)([x, x_26], **kwargs)
        x, fmap_26 = YOLOBlock(256, 3 * (5 + n_cls), 'y_26', trainable=True,
                               restore_output_weights=restore_output_weights)(x, **kwargs)
        x = Upsample(128, trainable=True)([x, x_52], **kwargs)
        x, fmap_52 = YOLOBlock(128, 3 * (5 + n_cls), 'y_52', trainable=True,
                               restore_output_weights=restore_output_weights)(x, **kwargs)

        return fmap_52, fmap_26, fmap_13
    return call


if __name__ == '__main__':
    inputs = Input(shape=(416, 416, 3))
    # model = yolo_v3(inputs)
    yolo_v3_net = YOLOv3Net(n_cls=80, restore_weights=True, trainable_backbone=True,
                            use_spp=True, restore_output_weights=True)
    model = Model(inputs, outputs=yolo_v3_net(inputs))
    model.save_weights('coco_init_weights_spp.h5')
    print(f'Restored {_weight_loader.cnt} of {len(_weight_loader.weights)} weights.')


