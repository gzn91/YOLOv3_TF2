import cv2
import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from anchors import CLASSES


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


def non_max_suppression(detections, confidence_threshold=0.2, iou_threshold=0.4, max_nr_boxes_per_cls=32):
    _, D = detections.shape

    bboxes, scores, classes = tf.split(detections, [4, 1, 1], axis=-1)
    scores = tf.squeeze(scores, axis=-1)
    classes = tf.squeeze(classes, axis=-1)
    found_cls, _ = tf.unique(classes)

    selected_scores = {}
    selected_boxes = {}

    for cls_idx in found_cls:
        cls_bool = tf.equal(classes, cls_idx)
        cls_boxes = tf.boolean_mask(bboxes, cls_bool, axis=0)
        cls_scores = tf.boolean_mask(scores, cls_bool, axis=0)

        inds = tf.image.non_max_suppression(cls_boxes, cls_scores,
                                            max_output_size=max_nr_boxes_per_cls,
                                            iou_threshold=iou_threshold,
                                            score_threshold=confidence_threshold)

        if tf.size(inds) > 0:
            best_boxes = tf.gather(cls_boxes, inds)
            best_scores = tf.gather(cls_scores, inds)

            selected_boxes[int(cls_idx)] = best_boxes
            selected_scores[int(cls_idx)] = best_scores

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


def read_image(path, target_size, return_full_res_image=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = target_size / max(img.shape)  # size ratio
    h, w = img.shape[:2]
    new_sz = (int(w * r), int(h * r))
    fx, fy = (target_size - np.array(new_sz)) // 2
    resized_img = cv2.resize(img, new_sz, interpolation=cv2.INTER_AREA)
    top = fy
    bottom = target_size - (fy + new_sz[1])
    left = fx
    right = target_size - (fx + new_sz[0])
    resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

    if return_full_res_image:
        return resized_img, np.r_[new_sz][::-1], fx / target_size, fy / target_size, img
    return resized_img, np.r_[new_sz][::-1], fx / target_size, fy / target_size


def resize_with_pad_tf(image, height, width, target_size):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    r = target_size / tf.math.maximum(width, height)  # size ratio
    new_sz = tf.cast([height * r, width * r], tf.int32)
    image = tf.image.resize(image, new_sz)

    pad_h = target_size - new_sz[0]
    pad_w = target_size - new_sz[1]
    pad_h_begin = tf.cast(pad_h / 2, tf.int32)
    pad_h_end = pad_h - pad_h_begin
    pad_w_begin = tf.cast(pad_w / 2, tf.int32)
    pad_w_end = pad_w - pad_w_begin

    image = tf.pad(image, [(pad_h_begin, pad_h_end), (pad_w_begin, pad_w_end), (0, 0)])
    return image, (pad_w_begin, pad_h_begin, tf.cast(new_sz, tf.float32))


def translate_bboxes(bboxes, pad_w, pad_h):
    pad_h = tf.cast(pad_h, tf.float32)
    pad_w = tf.cast(pad_w, tf.float32)
    xmin, ymin, xmax, ymax = tf.split(bboxes, num_or_size_splits=4, axis=-1)
    xmin = (pad_w + xmin)
    xmax = (pad_w + xmax)
    ymin = (pad_h + ymin)
    ymax = (pad_h + ymax)
    return tf.concat([xmin, ymin, xmax, ymax], axis=-1)


def draw_bbox(image, preds, bboxes, dw=0, dh=0, r=(1, 1), scale=True):
    CM = plt.cm.get_cmap('Set1')
    fig, ax = plt.subplots(1, figsize=(13, 7))
    scale_h, scale_w, _ = image.shape
    ax.imshow(image)
    for k, v in preds.items():
        for sc, bb in zip(preds[k], bboxes[k]):
            xmin, ymin, xmax, ymax = tf.split(bb, 4, -1)
            if scale:
                xmin = scale_w * (xmin - dw)
                xmax = scale_w * (xmax - dw)
                ymin = scale_h * (ymin - dh)
                ymax = scale_h * (ymax - dh)
            xmin *= r[1]
            xmax *= r[1]
            ymin *= r[0]
            ymax *= r[0]
            h = ymax - ymin
            w = xmax - xmin
            rec = Rectangle((xmin, ymin), width=w, height=h, fill=False, edgecolor=CM(k), lw=1)
            ax.add_patch(rec)
            ax.annotate(f'{CLASSES[k]}: {sc * 100:.2f}%', (xmin, ymin), fontsize=6, color='white')
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


def resize(image, size):
    height, width, _ = image.shape
    r = size / np.maximum(width, height)  # size ratio
    new_sz = np.array((width * r, height * r)).astype(np.int32)
    image = cv2.resize(image, tuple(new_sz), interpolation=cv2.INTER_AREA)

    return image, new_sz[::-1], r
