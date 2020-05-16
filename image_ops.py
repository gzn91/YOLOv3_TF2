import cv2
import numpy as np
import os
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import CLASSES


def compute_iou_np(box1, box2):
    # expand to broadcast
    box1 = np.expand_dims(box1, axis=0)
    box2 = np.expand_dims(box2, axis=-2)

    xymin = np.maximum(box1[..., :2], box2[..., :2])
    xymax = np.minimum(box1[..., 2:], box2[..., 2:])
    wh = np.maximum(xymax - xymin, 0.)

    inter_area = wh[..., 0] * wh[..., 1]
    box1_area = np.subtract(box1[..., 2], box1[..., 0]) * np.subtract(box1[..., 3], box1[..., 1])
    box2_area = np.subtract(box2[..., 2], box2[..., 0]) * np.subtract(box2[..., 3], box2[..., 1])
    outer_area = box1_area + box2_area - inter_area

    return np.max(inter_area / outer_area, axis=-1)


def compute_iou(box1, box2):
    # expand to broadcast
    box1 = tf.expand_dims(box1, axis=0)
    box2 = tf.expand_dims(box2, axis=-2)

    xymin = tf.math.maximum(box1[..., :2], box2[..., :2])
    xymax = tf.math.minimum(box1[..., 2:], box2[..., 2:])
    wh = tf.math.maximum(xymax - xymin, 0.)

    inter_area = wh[..., 0] * wh[..., 1]
    box1_area = tf.math.subtract(box1[..., 2], box1[..., 0]) * tf.math.subtract(box1[..., 3], box1[..., 1])
    box2_area = tf.math.subtract(box2[..., 2], box2[..., 0]) * tf.math.subtract(box2[..., 3], box2[..., 1])
    outer_area = box1_area + box2_area - inter_area

    return tf.reduce_max(inter_area / outer_area, axis=-1)


def non_max_suppression(y_pred, confidence_threshold=0.2, iou_threshold=0.4, max_nr_boxes=None):

    _, D = y_pred.shape

    conf_mask = y_pred[..., 4] > confidence_threshold
    y_pred = y_pred[conf_mask]
    if max_nr_boxes and y_pred.shape[0] > max_nr_boxes:
        _, inds = tf.math.top_k(y_pred[:, 4], k=max_nr_boxes)
        y_pred = tf.gather(y_pred, inds)
    bboxes, scores, preds = tf.split(y_pred, [4, 1, -1], axis=-1)
    scores = tf.squeeze(scores, axis=-1)
    classes = tf.argmax(preds, -1)
    found_cls, _ = tf.unique(classes)

    selected_scores = {}
    selected_boxes = {}

    for cls_idx in found_cls:
        selected_scores[int(cls_idx)] = []
        selected_boxes[int(cls_idx)] = []
        cls_mask = tf.equal(classes, cls_idx)
        cls_boxes = tf.boolean_mask(bboxes, cls_mask, axis=0)
        cls_scores = tf.boolean_mask(scores, cls_mask, axis=0)
        # sort by highest confidence
        idx_sorted = tf.argsort(cls_scores, axis=0)[::-1]
        cls_scores = tf.gather(cls_scores ,idx_sorted)
        cls_boxes = tf.gather(cls_boxes, idx_sorted)
        box_idx = tf.range(cls_boxes.shape[0])
        while len(box_idx) > 0:
            best_box = cls_boxes[box_idx[0]]
            best_score = cls_scores[box_idx[0]]
            other_boxes = tf.gather(cls_boxes, box_idx[1:])
            box_idx = box_idx[1:]
            selected_boxes[int(cls_idx)].append(best_box)
            selected_scores[int(cls_idx)].append(best_score)
            iou = compute_iou(tf.expand_dims(best_box, axis=0), other_boxes)
            iou_mask = iou < iou_threshold
            box_idx = box_idx[iou_mask]

    return selected_scores, selected_boxes


def non_max_suppression_np(y_pred, confidence_threshold=0.2, iou_threshold=0.4):

    _, D = y_pred.shape
    conf_mask = y_pred[..., 4] > confidence_threshold
    y_pred = y_pred[conf_mask].reshape(1,-1, D)
    y_pred = y_pred[:,np.argsort(y_pred[..., 4])[-40:]]
    bboxes, scores, preds = np.split(y_pred, [4,5], axis=-1)
    scores = np.squeeze(scores, axis=-1)
    classes = preds.argmax(-1)

    selected_scores = {}
    selected_boxes = {}

    # iterate over batch
    for i, (score, cls, bbox) in enumerate(zip(scores, classes, bboxes)):

        for cls_idx in np.unique(cls):
            selected_scores[cls_idx] = []
            selected_boxes[cls_idx] = []
            cls_mask = cls == cls_idx
            cls_boxes = bbox[cls_mask]
            cls_scores = score[cls_mask]

            # sort by highest confidence
            idx_sorted = np.argsort(cls_scores, axis=-1)[::-1]
            cls_scores = cls_scores[idx_sorted]
            cls_boxes = cls_boxes[idx_sorted]
            box_idx = np.arange(cls_boxes.shape[0])
            while len(box_idx) > 0:
                best_box = cls_boxes[box_idx[0]]
                best_score = cls_scores[box_idx[0]]
                other_boxes = cls_boxes[box_idx[1:]]
                box_idx = box_idx[1:]
                selected_boxes[cls_idx].append(best_box)
                selected_scores[cls_idx].append(best_score)
                iou = compute_iou_np(np.expand_dims(best_box, axis=0), other_boxes)
                iou_mask = iou < iou_threshold
                box_idx = box_idx[iou_mask]

    return selected_scores, selected_boxes


def read_image(path, img=None, return_full_res_image=False):
    if img is None:
        print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = 416 / max(img.shape)  # size ratio
    h, w = img.shape[:2]
    new_sz = (max(int(w * r), 208), max(int(h * r), 208))
    fx, fy = (416 - np.array(new_sz)) // 2
    resized_img = cv2.resize(img, new_sz, interpolation=cv2.INTER_AREA)
    top = fy
    bottom = 416 - (fy + new_sz[1])
    left = fx
    right = 416 - (fx + new_sz[0])
    resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

    if return_full_res_image:
        return resized_img, new_sz, fx / 416, fy / 416, img
    return resized_img, new_sz, fx/416, fy/416


def draw_bbox(image, preds, bboxes, dw=0, dh=0, new_size=(416, 416)):
    CM = plt.cm.get_cmap('Set1')
    fig, ax = plt.subplots(1, figsize=(13, 7))
    scale_h, scale_w, _ = image.shape
    ax.imshow(image)
    r = np.array([416, 416]) / np.array(new_size)
    for k, v in preds.items():
        for sc, bb in zip(preds[k], bboxes[k]):
            xmin, ymin, xmax, ymax = tf.split(bb, 4, -1)
            xmin = scale_w * (xmin - dw) * r[0]
            xmax = scale_w * (xmax - dw) * r[0]
            ymin = scale_h * (ymin - dh) * r[1]
            ymax = scale_h * (ymax - dh) * r[1]
            h = ymax - ymin
            w = xmax - xmin

            rec = Rectangle((xmin, ymin), width=w, height=h, fill=False, edgecolor=CM(k), lw=2)
            ax.add_patch(rec)
            ax.annotate(f'{CLASSES[k]}: {sc * 100:.2f}%', (xmin, ymin), fontsize=8, color='white')
            ax.axis('off')

    return fig


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def get_img_from_fig(fig, dpi=180):

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
