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

    def __init__(self):
        self.cnt = 0
        with open('yolov3.weights', 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
            self.weights = np.fromfile(wf, dtype=np.float32)

    def __call__(self, num_weights):
        weights = self.weights[self.cnt:self.cnt + num_weights]
        self.cnt += int(num_weights)
        return weights


class DarknetConv2D(Layer):
    _weight_loader = WeightLoader()
    restore = False
    trainable = True

    def __init__(self, filters=128, kernel_size=(3, 3), strides=(1, 1), pad='same', use_bn=True, skip_weights=False,
                 trainable=None, dtype=None):
        super(DarknetConv2D, self).__init__(dtype=dtype)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad
        self.use_bn = use_bn
        self.skip_weights = skip_weights
        if trainable:
            self.trainable = trainable
        else:
            self.trainable = DarknetConv2D.trainable

    def build(self, input_shape):
        conv_weights = None
        bn_weights = None
        filters = self.filters
        self.ops = []
        if DarknetConv2D.restore:

            if self.skip_weights:
                filters = 3 * (5 + 80)

            conv_shape = (filters, input_shape[-1], self.kernel_size[0], self.kernel_size[1])

            if self.use_bn:
                bn_weights = DarknetConv2D._weight_loader(4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                conv_weights = DarknetConv2D._weight_loader(np.prod(conv_shape))
                conv_weights = [conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])]

            else:
                conv_bias = DarknetConv2D._weight_loader(filters)
                conv_weights = DarknetConv2D._weight_loader(np.prod(conv_shape))
                conv_weights = [conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0]), conv_bias]

            if self.skip_weights:
                conv_weights = None
                bn_weights = None

        self.ops.append(Conv2D(self.filters, self.kernel_size,
                               padding=self.pad,
                               strides=self.strides,
                               activation='linear',
                               kernel_regularizer=l2(5e-4),
                               use_bias=not self.use_bn,
                               weights=conv_weights,
                               trainable=self.trainable,
                               dtype=self.dtype))
        if self.use_bn:
            self.ops.append(BatchNormalization(weights=bn_weights, trainable=self.trainable, dtype=self.dtype))
            self.ops.append(LeakyReLU(alpha=0.1, dtype=self.dtype))
        return self.ops

    def call(self, inputs, **kwargs):
        x = inputs
        for op in self.ops:
            x = op(x, **kwargs)
        return x


class DarknetResidual(Layer):

    def __init__(self, filters):
        super(DarknetResidual, self).__init__()

        self.conv1 = DarknetConv2D(filters // 2, (1, 1))
        self.conv2 = DarknetConv2D(filters, (3, 3))
        self.add = Add()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs, **kwargs)
        x = self.conv2(x, **kwargs)
        x = self.add([inputs, x])
        return x


class SPPLayer(Layer):

    def __init__(self):
        super(SPPLayer, self).__init__()
        self.maxpool_5 = MaxPool2D(pool_size=(5, 5), strides=1, padding='same')
        self.maxpool_9 = MaxPool2D(pool_size=(9, 9), strides=1, padding='same')
        self.maxpool_13 = MaxPool2D(pool_size=(13, 13), strides=1, padding='same')
        self.concat = Concatenate()

    def call(self, inputs, **kwargs):
        x_5 = self.maxpool_5(inputs)
        x_9 = self.maxpool_9(inputs)
        x_13 = self.maxpool_13(inputs)
        x = self.concat([x_13, x_9, x_5, inputs])
        return x


class YOLOBlock(Layer):

    def __init__(self, filters, output_filters, name, use_spp=False):
        super(YOLOBlock, self).__init__()
        self.filters = filters
        self.output_filters = output_filters
        self._name = name
        self.use_spp = use_spp
        self.ops = []
        self.output_ops = []

    def build(self, input_shape):
        self.ops.append(DarknetConv2D(self.filters, (1, 1)))
        self.ops.append(DarknetConv2D(self.filters * 2, (3, 3)))
        self.ops.append(DarknetConv2D(self.filters, (1, 1)))
        if self.use_spp:
            self.ops.append(SPPLayer())
            self.ops.append(DarknetConv2D(self.filters, (1, 1)))
        self.ops.append(DarknetConv2D(self.filters * 2, (3, 3)))
        self.ops.append(DarknetConv2D(self.filters, (1, 1)))
        self.output_ops.append(DarknetConv2D(self.filters * 2, (3, 3)))
        self.output_ops.append(
            DarknetConv2D(self.output_filters, (1, 1), use_bn=False, skip_weights=False, trainable=True, dtype=tf.float32))
        self.output_ops.append(Reshape([*input_shape[1:-1], 3, self.output_filters // 3], name=self._name, dtype=tf.float32))

    def call(self, inputs, **kwargs):
        x = inputs
        for op in self.ops:
            x = op(x, **kwargs)

        fmap = x
        for fmap_op in self.output_ops:
            fmap = fmap_op(fmap, **kwargs)

        return x, fmap


class DarknetBlock(Layer):

    def __init__(self, filters, n_blocks):
        super(DarknetBlock, self).__init__()

        self.ops = []

        self.ops.append(ZeroPadding2D(((1, 0), (1, 0))))
        self.ops.append(DarknetConv2D(filters, (3, 3), strides=(2, 2), pad='valid'))
        for i in range(n_blocks):
            self.ops.append(DarknetResidual(filters))

    def call(self, inputs, **kwargs):
        x = inputs
        for op in self.ops:
            x = op(x, **kwargs)

        return x


class Darknet53Body(Layer):

    def __init__(self):
        super(Darknet53Body, self).__init__()
        self.darknet_blocks = []
        self.filters_per_level = [64, 128, 256, 512, 1024]
        self.n_blocks_per_level = [1, 2, 8, 8, 4]

    def build(self, input_shape):

        self.conv = DarknetConv2D(32, (3, 3))
        for filters, n_blocks in zip(self.filters_per_level, self.n_blocks_per_level):
            self.darknet_blocks.append(DarknetBlock(filters, n_blocks))

    def call(self, inputs, **kwargs):

        x = self.conv(inputs)

        fmaps = []
        for darknet_block in self.darknet_blocks:
            x = darknet_block(x, **kwargs)
            fmaps.append(x)

        x_52, x_26, x_13 = fmaps[-3:]

        return x_52, x_26, x_13


class Upsample(Layer):

    def __init__(self, filters):
        super(Upsample, self).__init__()
        self.conv = DarknetConv2D(filters, (1, 1))
        self.upsample = UpSampling2D(2)
        self.concat = Concatenate()

    def call(self, inputs, **kwargs):
        x, lateral_x = inputs

        x = self.conv(x, **kwargs)
        x = self.upsample(x)
        x = self.concat([x, lateral_x])

        return x


class YOLOv3Net(Layer):

    def __init__(self, restore_weights=True, ncls=80, train_backbone=False):
        super(YOLOv3Net, self).__init__()
        self.restore = restore_weights
        self._ncls = ncls
        self._train_backbone = train_backbone
        if self.restore:
            DarknetConv2D.restore = True

    # noinspection PyAttributeOutsideInit
    # pylint: disable=W0201
    def build(self, input_shape):

        if self.restore:
            print('Restoring weights...')


        self.darknet = Darknet53Body()


        self.yolo_block_13 = YOLOBlock(512, 3 * (5 + self._ncls), 'y_pred_13')
        self.upsample_26 = Upsample(256)
        self.yolo_block_26 = YOLOBlock(256, 3 * (5 + self._ncls), 'y_pred_26')
        self.upsample_52 = Upsample(128)
        self.yolo_block_52 = YOLOBlock(128, 3 * (5 + self._ncls), 'y_pred_52')

    def call(self, inputs, **kwargs):
        if not self._train_backbone:
            DarknetConv2D.trainable = False
        x_52, x_26, x_13 = self.darknet(inputs, **kwargs)
        DarknetConv2D.trainable = True

        x, fmap_13 = self.yolo_block_13(x_13, **kwargs)
        x = self.upsample_26([x, x_26], **kwargs)
        x, fmap_26 = self.yolo_block_26(x, **kwargs)

        x = self.upsample_52([x, x_52], **kwargs)
        x, fmap_52 = self.yolo_block_52(x, **kwargs)

        return fmap_52, fmap_26, fmap_13


if __name__ == '__main__':
    inputs = Input(shape=(1024, 1024, 3))
    # model = yolo_v3(inputs)
    yolo_v3_net = YOLOv3Net(ncls=80, restore_weights=True)
    model = Model(inputs, outputs=yolo_v3_net(inputs))
    model.save_weights('coco_init_yolov3_weights.h5')
    print(f'Restored {len(DarknetConv2D._weight_loader.weights)} of {DarknetConv2D._weight_loader.cnt} weights.')
    # load_darknet_weights(model,'yolov3.weights')



