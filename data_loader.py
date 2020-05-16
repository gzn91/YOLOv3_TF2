import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from utils import AnchorBoxes, CLASSES
from image_ops import non_max_suppression, draw_bbox
from augmentor import Augmentor



class DataLoader(object):

    def __init__(self, record_path, n_cls, img_shape):
        self.path = record_path
        self._ncls = n_cls
        self.img_shape = img_shape
        self.desired_size = img_shape[0]
        self.encoder = AnchorBoxes(img_size=self.desired_size, ncls=self._ncls)
        self.augmentor = Augmentor()

    def resize_with_pad(self, image, width, height):
        r = self.desired_size / tf.math.maximum(width, height)  # size ratio
        new_sz = tf.cast([height * r, width * r], tf.int32)
        image = tf.image.resize(image, new_sz)
        # TODO: Here

        pad_h = self.desired_size - new_sz[0]
        pad_w = self.desired_size - new_sz[1]
        pad_h_begin = tf.cast(pad_h / 2, tf.int32)
        pad_h_end = pad_h - pad_h_begin
        pad_w_begin = tf.cast(pad_w / 2, tf.int32)
        pad_w_end = pad_w - pad_w_begin

        image = tf.pad(image, [(pad_h_begin, pad_h_end), (pad_w_begin, pad_w_end), (0, 0)])
        return image, (pad_w_begin, pad_h_begin, tf.cast(new_sz, tf.float32))

    def translate_bboxes(self, bboxes, pad_w, pad_h, width, height):
        pad_h = tf.cast(pad_h, tf.float32)
        pad_w = tf.cast(pad_w, tf.float32)
        xmin, ymin, xmax, ymax = tf.split(bboxes, num_or_size_splits=4, axis=-1)
        xmin = (pad_w + width * xmin)
        xmax = (pad_w + width * xmax)
        ymin = (pad_h + height * ymin)
        ymax = (pad_h + height * ymax)
        return tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    def create_dataset(self, batch_size):

        def _parse_fn(tf_record, training):
            features = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string),
                'filename': tf.io.FixedLenFeature([], tf.string),
                'class': tf.io.VarLenFeature(tf.string),
                'xmin': tf.io.VarLenFeature(tf.float32),
                'xmax': tf.io.VarLenFeature(tf.float32),
                'ymin': tf.io.VarLenFeature(tf.float32),
                'ymax': tf.io.VarLenFeature(tf.float32),
                'label': tf.io.VarLenFeature(tf.int64)
            }

            features = tf.io.parse_single_example(tf_record, features)
            filename = features['filename']
            image = tf.io.decode_jpeg(features['image'])
            image = tf.cast(image, tf.float32)
            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            image = tf.reshape(image, (height, width, 3))
            height = tf.cast(height, tf.float32)
            width = tf.cast(width, tf.float32)
            image, (pad_w, pad_h, new_sz) = self.resize_with_pad(image, width, height)
            height, width = tf.unstack(new_sz, 2, axis=-1)

            xmin = tf.cast(features['xmin'].values, tf.float32)
            xmax = tf.cast(features['xmax'].values, tf.float32)
            ymin = tf.cast(features['ymin'].values, tf.float32)
            ymax = tf.cast(features['ymax'].values, tf.float32)
            bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            bboxes = self.translate_bboxes(bboxes, pad_w, pad_h, width, height)
            xmin, ymin, xmax, ymax = tf.unstack(bboxes, 4, axis=-1)
            bboxes = tf.stack([xmin, ymin, (xmax - xmin), (ymax - ymin)], axis=-1) / self.desired_size
            labels = tf.cast(features['label'].values, tf.int64)


            if training:
                image, bboxes, labels = tf.numpy_function(self.augmentor.augment, [image, bboxes, labels],
                                                   [tf.float32, tf.float32, tf.int64])

            y_52, y_26, y_13 = tf.numpy_function(self.encoder.encode, [labels, bboxes], [tf.float32, tf.float32, tf.float32])

            return image/255., {'y_52': y_52, 'y_26': y_26, 'y_13': y_13}, filename

        train_ds = os.path.join(self.path, 'training.tfrecord')
        val_ds = os.path.join(self.path, 'validation.tfrecord')

        # train_ds = os.path.join(data_path, 'training-*.tfrecord')
        # val_ds = os.path.join(data_path, 'validation-*.tfrecord')

        def generate_tfdataset(paths, batch_size=2, training=True, repeat=1):
            dataset = tf.data.TFRecordDataset(paths)
            dataset = dataset.map(lambda x: _parse_fn(x, training=training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(16).repeat(repeat)
            dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

            # dataset = tf.data.Dataset.list_files(paths)
            # dataset = dataset.shuffle(10).repeat()
            # dataset = dataset.interleave(tf.data.TFRecordDataset,
            #                              cycle_length=10, block_length=1, num_parallel_calls=10)
            # dataset = dataset.map(lambda x: _parse_fn(x, training=training), num_parallel_calls=16).prefetch(
            #     tf.contrib.data.AUTOTUNE)
            # dataset = dataset.shuffle(512).batch(batch_size)
            return dataset

        return generate_tfdataset(train_ds, batch_size=batch_size, repeat=8*50), generate_tfdataset(val_ds, batch_size=4, training=False, repeat=30)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    loader = DataLoader('./data/', n_cls=21, img_shape=(1024, 1024, 3))
    train_set, val_set = loader.create_dataset()

    for imgs, labels, fnames in train_set:
        y_52 = labels['y_52']
        y_26 = labels['y_26']
        y_13 = labels['y_13']
        y_52 = loader.encoder.decode_gt(y_52)
        y_26 = loader.encoder.decode_gt(y_26)
        y_13 = loader.encoder.decode_gt(y_13)

        y = np.concatenate([y_52, y_26, y_13], axis=1)

        selected_scores = []
        selected_boxes = []
        for j, p in enumerate(y):
            scores, bboxes = non_max_suppression(p, 0.2, iou_threshold=0.4)
            selected_scores.append(scores)
            selected_boxes.append(bboxes)
            print(bboxes)

            f = draw_bbox(imgs[j], scores, bboxes)
            plt.show()





