import tensorflow as tf
from anchors import AnchorBoxes
from layers import YOLOv3Net
from losses import YoloLoss


class YOLOv3(object):

    def __init__(self, img_shape=(416,416,3), ncls=80, use_pretrained_weights=True, use_spp=True):
        self._img_shape = img_shape
        self._img_size = img_shape[0]
        self._ncls = ncls

        self.anchor_manipulator = AnchorBoxes(img_size=img_shape[0], ncls=ncls)
        self.anchors = self.anchor_manipulator.anchors

        self.input_ = tf.keras.Input(img_shape, name='input')
        self.network = tf.keras.Model(inputs=self.input_, outputs=YOLOv3Net(restore_weights=False, n_cls=self._ncls,
                                                                            trainable_backbone=True,
                                                                            use_spp=use_spp)(self.input_))

        def _decode(inputs, max_to_keep=2000):
            y_52, y_26, y_13 = inputs
            y_52 = self.anchor_manipulator.decode(y_52, [0, 1, 2])
            y_26 = self.anchor_manipulator.decode(y_26, [3, 4, 5])
            y_13 = self.anchor_manipulator.decode(y_13, [6, 7, 8])
            y = tf.concat([y_52, y_26, y_13], axis=1)
            boxes, obj_score, cls_scores = tf.split(y,  [4, 1, -1], -1)
            cls_ids = tf.cast(tf.argmax(cls_scores, axis=-1)[..., tf.newaxis], tf.float32)

            inds = tf.nn.top_k(obj_score[..., 0], k=max_to_keep).indices

            obj_score = tf.gather(obj_score, inds, batch_dims=1, axis=1)
            cls_ids = tf.gather(cls_ids, inds, batch_dims=1, axis=1)
            boxes = tf.gather(boxes, inds, batch_dims=1, axis=1)
            preds = tf.concat([boxes, obj_score, cls_ids], axis=-1)

            return preds

        y_52, y_26, y_13 = self.network.outputs
        self.serving = tf.keras.Model(inputs=self.input_,
                                      outputs=_decode([y_52, y_26, y_13]))
        if use_pretrained_weights:
            weight_path = 'coco_init_weights_spp.h5' if use_spp else 'coco_init_weights.h5'
            self.network.load_weights(weight_path)  # run layers to create this

        loss_52 = YoloLoss([0, 1, 2],
                           self._img_size//8,
                           ncls=ncls,
                           img_shape=img_shape)
        loss_26 = YoloLoss([3, 4, 5],
                           self._img_size//16,
                           ncls=ncls,
                           img_shape=img_shape)
        loss_13 = YoloLoss([6, 7, 8],
                           self._img_size//32,
                           ncls=ncls,
                           img_shape=img_shape)

        self.loss = {'y_52': lambda y_true, y_pred: loss_52(y_true, y_pred),
                     'y_26': lambda y_true, y_pred: loss_26(y_true, y_pred),
                     'y_13': lambda y_true, y_pred: loss_13(y_true, y_pred)}

    def __call__(self, *inputs, **kwargs):
        return self.serving(inputs, **kwargs)

    def compile(self, optimizer):
        self.network.compile(optimizer=optimizer, loss=self.loss)

    def fit(self, train_ds, steps_per_epoch=100, validation_data=None,
            validation_steps=20, epochs=1000, callbacks=None,
            initial_epoch=0):
        self.network.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
                         validation_steps=validation_steps, epochs=epochs, callbacks=callbacks,
                         initial_epoch=initial_epoch, shuffle=False)

    def load_weights(self, path, expect_partial=False):
        if expect_partial:
            self.network.load_weights(path).expect_partial()

        else:
            self.network.load_weights(path)

