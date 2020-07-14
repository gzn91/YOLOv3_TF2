import tensorflow as tf
import os
from typing import List, Dict, Tuple, Any, Union
import numpy as np
from anchors import AnchorBoxes
from image_ops import non_max_suppression, draw_bbox, resize_with_pad_tf, translate_bboxes
from augmentor import Augmentor

MAX_GT_PER_IMAGE = 500

class DataLoader(object):

    def __init__(self, record_path, n_cls, img_shape):
        self.path = record_path
        self._ncls = n_cls
        self.img_shape = img_shape
        self.desired_size = img_shape[0]
        self.encoder = AnchorBoxes(img_size=self.desired_size, ncls=self._ncls)
        self.augmentor = Augmentor()

    def create_dataset(self, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        def _parse_fn(tf_record: tf.Tensor,
                      training: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            features = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string),
                'class': tf.io.VarLenFeature(tf.string),
                'xmin': tf.io.VarLenFeature(tf.float32),
                'xmax': tf.io.VarLenFeature(tf.float32),
                'ymin': tf.io.VarLenFeature(tf.float32),
                'ymax': tf.io.VarLenFeature(tf.float32),
                'label': tf.io.VarLenFeature(tf.int64)
            }

            features = tf.io.parse_single_example(tf_record, features)
            image = tf.io.decode_jpeg(features['image'])
            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            image = tf.reshape(image, (height, width, 3))
            image, (pad_w, pad_h, new_sz) = resize_with_pad_tf(image, height, width, target_size=self.desired_size)
            height, width = tf.unstack(new_sz, 2, axis=-1)

            xmin = tf.cast(features['xmin'].values, tf.float32) * width
            xmax = tf.cast(features['xmax'].values, tf.float32) * width
            ymin = tf.cast(features['ymin'].values, tf.float32) * height
            ymax = tf.cast(features['ymax'].values, tf.float32) * height

            bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            bboxes = translate_bboxes(bboxes, pad_w, pad_h)

            labels = tf.cast(features['label'].values, tf.int64)
            if self._ncls == 1:
                labels = tf.zeros_like(labels)

            if training:
                image = tf.cast(image, tf.uint8)
                image, bboxes, labels = tf.numpy_function(self.augmentor.augment, [image, bboxes, labels],
                                                   [tf.float32, tf.float32, tf.int64])

            bboxes = tf.reshape(bboxes, (-1, 4))
            labels = tf.reshape(labels, (-1,))
            image = tf.reshape(image, (self.desired_size, self.desired_size, 3))

            bboxes = bboxes / self.desired_size

            y_52, y_26, y_13 = tf.numpy_function(self.encoder.encode, [labels, bboxes], [tf.float32, tf.float32, tf.float32])

            n_objs = tf.size(labels)
            padding = MAX_GT_PER_IMAGE - n_objs

            # pad boxes and labels
            labels = tf.pad(labels, [(0, padding)])
            bboxes = tf.pad(bboxes, [(0, padding), (0, 0)])

            y_52.set_shape((52, 52, 3, 5 + self._ncls))
            y_26.set_shape((26, 26, 3, 5 + self._ncls))
            y_13.set_shape((13, 13, 3, 5 + self._ncls))

            return {'input': image/255, 'gt_boxes': bboxes, 'gt_cls': labels}, {'y_52': y_52, 'y_26': y_26, 'y_13': y_13}

        train_ds = os.path.join(self.path, 'training-*.tfrecord')
        val_ds = os.path.join(self.path, 'validation-*.tfrecord')

        def generate_tfdataset(paths: str, batch_size: int = 2, buffer_size: int = 10, training: bool = True):
            dataset = tf.data.Dataset.list_files(paths)
            dataset = dataset.shuffle(buffer_size).repeat()
            dataset = dataset.interleave(tf.data.TFRecordDataset, block_length=1, num_parallel_calls=-1)
            dataset = dataset.map(lambda x: _parse_fn(x, training=training), num_parallel_calls=-1)
            dataset = dataset.shuffle(512).batch(batch_size).prefetch(-1)
            return dataset

        return (generate_tfdataset(train_ds,
                                  buffer_size=10,
                                  batch_size=batch_size),
                generate_tfdataset(val_ds,
                                  batch_size=4,
                                  training=False))


if __name__=='__main__':
    import matplotlib.pyplot as plt
    loader = DataLoader('records', n_cls=80, img_shape=(416, 416, 3))
    train_set, val_set = loader.create_dataset(2)

    for inputs, labels in train_set:
        imgs = inputs['input']
        y_52 = labels['y_52']
        y_26 = labels['y_26']
        y_13 = labels['y_13']


        def _decode(inputs, max_to_keep=2000):
            y_52, y_26, y_13 = inputs
            y_52 = loader.encoder.decode_gt(y_52)
            y_26 = loader.encoder.decode_gt(y_26)
            y_13 = loader.encoder.decode_gt(y_13)
            y = tf.concat([y_52, y_26, y_13], axis=1)
            boxes, obj_score, cls_scores = tf.split(y, [4, 1, -1], -1)
            cls_ids = tf.cast(tf.argmax(cls_scores, axis=-1)[..., tf.newaxis], tf.float32)

            inds = tf.nn.top_k(obj_score[..., 0], k=max_to_keep).indices

            obj_score = tf.gather(obj_score, inds, batch_dims=1, axis=1)
            cls_ids = tf.gather(cls_ids, inds, batch_dims=1, axis=1)
            boxes = tf.gather(boxes, inds, batch_dims=1, axis=1)

            preds = tf.concat([boxes, obj_score, cls_ids], axis=-1)

            return preds

        y = _decode([y_52, y_26, y_13])

        selected_scores = []
        selected_boxes = []
        for j, p in enumerate(y):
            scores, bboxes = non_max_suppression(p, 0.2, iou_threshold=0.4)
            selected_scores.append(scores)
            selected_boxes.append(bboxes)

            f = draw_bbox(imgs[j], scores, bboxes)
            plt.show()





