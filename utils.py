import cv2
import numpy as np
import os
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CLASSES = ['person',
           'bicycle',
           'car',
           'motorbike',
           'aeroplane',
           'bus',
           'train',
           'truck',
           'boat',
           'traffic light',
           'fire hydrant',
           'stop sign',
           'parking meter',
           'bench',
           'bird',
           'cat',
           'dog',
           'horse',
           'sheep',
           'cow',
           'elephant',
           'bear',
           'zebra',
           'giraffe',
           'backpack',
           'umbrella',
           'handbag',
           'tie',
           'suitcase',
           'frisbee',
           'skis',
           'snowboard',
           'sports ball',
           'kite',
           'baseball bat',
           'baseball glove',
           'skateboard',
           'surfboard',
           'tennis racket',
           'bottle',
           'wine glass',
           'cup',
           'fork',
           'knife',
           'spoon',
           'bowl',
           'banana',
           'apple',
           'sandwich',
           'orange',
           'broccoli',
           'carrot',
           'hot dog',
           'pizza',
           'donut',
           'cake',
           'chair',
           'sofa',
           'pottedplant',
           'bed',
           'diningtable',
           'toilet',
           'tvmonitor',
           'laptop',
           'mouse',
           'remote',
           'keyboard',
           'cell phone',
           'microwave',
           'oven',
           'toaster',
           'sink',
           'refrigerator',
           'book',
           'clock',
           'vase',
           'scissors',
           'teddy bear',
           'hair drier',
           'toothbrush']

LABEL_MAP = {'person': 0, 'bicycle': 1, 'car': 2, 'motorbike': 3, 'aeroplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
             'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
             'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
             'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
             'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
             'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39,
             'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47,
             'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54,
             'cake': 55, 'chair': 56, 'sofa': 57, 'pottedplant': 58, 'bed': 59, 'diningtable': 60, 'toilet': 61,
             'tvmonitor': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67,
             'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74,
             'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}


# with open('classes.txt', 'rb') as f:
#     CLASSES = [row.decode('utf-8').strip() for row in f]
# print(CLASSES)


def compute_iou_wh(boxes, anchors):
    anchors = np.expand_dims(anchors, 0)
    boxes = np.expand_dims(boxes, -2)

    inter_wh = np.minimum(boxes, anchors)
    box_area = np.prod(boxes, axis=-1)  # (n,k)
    anchor_area = np.prod(anchors, axis=-1)  # (n,k)

    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    outer_area = box_area + anchor_area - inter_area

    return inter_area / outer_area


class AnchorBoxes(object):

    def __init__(self, img_size, filename='anchors.txt', ncls=80):
        anchors = np.genfromtxt(filename, np.float32, delimiter=',')

        # order
        idx = np.argsort(anchors.prod(axis=-1))
        self.anchors = anchors[idx] * img_size / 416
        self.img_size = img_size
        self.ncls = ncls

    def encode(self, labels, boxes):
        mb_size = labels.shape[0]

        def _encode(box, anchor_idx, anchor_mask, grid_size):
            y_true = np.zeros((grid_size, grid_size, 3, 5 + self.ncls), dtype=np.float32)

            mask = (anchor_idx[:, None] == np.tile(anchor_mask, (anchor_idx.size, 1))).sum(axis=-1) == 1
            box = box[mask]
            label = labels[mask]
            anchor_idx_ = anchor_idx[mask] - np.min(anchor_mask)
            x = np.floor(box[:, 0] * grid_size).astype(np.int32)
            y = np.floor(box[:, 1] * grid_size).astype(np.int32)
            y_true[y, x, anchor_idx_, :4] = box
            y_true[y, x, anchor_idx_, 4] = 1.0
            y_true[y, x, anchor_idx_, 5 + label] = 1.0

            return y_true

        # convert from x_left, x_right,w,h to cx,cy,w,h
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

        iou = compute_iou_wh(boxes[:, 2:] * self.img_size, self.anchors)
        best_anchors = np.argmax(iou, axis=-1)

        # small objects
        y_true_32 = _encode(boxes, best_anchors, np.array([0, 1, 2]), grid_size=self.img_size // 8)
        y_true_16 = _encode(boxes, best_anchors, np.array([3, 4, 5]), grid_size=self.img_size // 16)
        y_true_8 = _encode(boxes, best_anchors, np.array([6, 7, 8]), grid_size=self.img_size // 32)

        return y_true_32, y_true_16, y_true_8

    def decode(self, y_pred, anchor_mask):
        B, G, _, A, D = y_pred.get_shape()

        anchor = self.anchors[anchor_mask]

        x, y, w, h, obj_mask, labels = tf.split(y_pred, num_or_size_splits=[1, 1, 1, 1, 1, -1], axis=-1)

        obj_mask = tf.nn.sigmoid(obj_mask)
        labels = tf.nn.sigmoid(labels)

        # create grids and calculate xy
        x_grid = tf.tile(tf.reshape(tf.range(G, dtype=tf.float32), [1, 1, -1, 1, 1]),
                         [1, G, 1, 1, 1])
        y_grid = tf.tile(tf.reshape(tf.range(G, dtype=tf.float32), [1, -1, 1, 1, 1]),
                         [1, 1, G, 1, 1])

        x = (tf.nn.sigmoid(x) + x_grid) / G
        y = (tf.nn.sigmoid(y) + y_grid) / G

        anchor = tf.cast(tf.reshape(anchor, [1, 1, 1, 3, 2]) / self.img_size, tf.float32)
        anchor_w, anchor_h = tf.split(anchor, 2, axis=-1)
        w = tf.math.exp(w) * anchor_w
        h = tf.math.exp(h) * anchor_h

        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2

        y_pred = tf.reshape(tf.concat([xmin, ymin, xmax, ymax, obj_mask, labels], axis=-1), (B, -1, 5 + self.ncls))

        return y_pred

    def decode_gt(self, y_pred):
        B, G, _, A, D = y_pred.get_shape()

        x, y, w, h, obj_mask, labels = tf.split(y_pred, num_or_size_splits=[1, 1, 1, 1, 1, -1], axis=-1)

        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2

        y_pred = tf.reshape(tf.concat([xmin, ymin, xmax, ymax, obj_mask, labels], axis=-1), (B, -1, 5 + self.ncls))

        return y_pred
