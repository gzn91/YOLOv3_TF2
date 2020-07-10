import numpy as np
from data_loader import DataLoader

import tensorflow as tf
import os

def compute_iou_wh(boxes, anchors):
    anchors = np.expand_dims(anchors, 0)
    boxes = np.expand_dims(boxes, -2)

    inter_wh = np.minimum(boxes, anchors)
    box_area = np.prod(boxes, axis=-1)  # (n,k)
    anchor_area = np.prod(anchors, axis=-1)  # (n,k)

    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    outer_area = box_area + anchor_area - inter_area

    return inter_area / outer_area

class Kmeans(object):
    def __init__(self, record_path, n_cls, img_shape):
        self.path = record_path
        self._ncls = n_cls
        self.img_shape = img_shape
        self.desired_size = img_shape[0]



class KMeans():

    def __init__(self, n_clusters, img_size=416, record_path=None):
        self.path = record_path
        self._n_clusters = n_clusters
        self.centers = np.random.rand(self._n_clusters,2)
        self.img_size = img_size

    def create_bboxes(self):
        def _parse_fn(tf_record):
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

            xmin = tf.cast(features['xmin'].values, tf.float32)
            xmax = tf.cast(features['xmax'].values, tf.float32)
            ymin = tf.cast(features['ymin'].values, tf.float32)
            ymax = tf.cast(features['ymax'].values, tf.float32)
            bboxes = tf.stack([xmin, ymin, (xmax - xmin), (ymax - ymin)], axis=-1)

            return bboxes

        dataset = tf.data.Dataset.list_files(os.path.join(self.path, 'training-*.tfrecord'))
        dataset = dataset.interleave(tf.data.TFRecordDataset, block_length=1, num_parallel_calls=-1)
        dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1)

        bboxes = []
        for bbox in dataset:
            bboxes.append(bbox[0])

        return np.concatenate(bboxes, 0)

    def fit(self, boxes):

        center_idx = np.random.choice(np.arange(boxes.shape[0], dtype='int32'), self._n_clusters, replace=False)
        centers = boxes[center_idx]

        last_labels = np.zeros(boxes.shape[0])
        current_labels = np.ones_like(last_labels)

        while not (last_labels == current_labels).all():
            last_labels = current_labels
            d = 1 - compute_iou_wh(boxes, centers)
            current_labels = np.argmin(d, -1)

            for i in range(self._n_clusters):
                if len(boxes[current_labels==i]) != 0:
                    centers[i] = np.median(boxes[current_labels==i], out=centers[i], axis=0)
                else:
                    centers[i] = np.random.rand(2,)
            print('acc:',np.sum(last_labels == current_labels)/last_labels.size)

        return np.int32(centers*self.img_size)


if __name__=='__main__':

    km = KMeans(9, img_size=416, record_path='records')
    bboxes = km.create_bboxes()
    print(bboxes.shape)
    centers = km.fit(bboxes[:,2:])
    print(centers)
    np.savetxt('anchors2.csv', np.int32(centers), fmt='%i', delimiter=',')