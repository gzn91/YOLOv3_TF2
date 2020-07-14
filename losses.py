from typing import List, Tuple
import tensorflow as tf
from anchors import AnchorBoxes
from image_ops import compute_iou


class YoloLoss(object):

    def __init__(self,
                 anchors_idx: List[int],
                 grid_size: int,
                 ncls: int = 80,
                 img_shape: Tuple[int, int, int] = (416, 416, 3),
                 lambda_coord: int = 2.0):

        self.anchor_idx = anchors_idx
        self.grid_size = grid_size
        self.ncls = ncls
        self.img_shape = img_shape
        self.lambda_coord = lambda_coord
        self.encoder = AnchorBoxes(img_size=img_shape[0], ncls=ncls)
        self.anchor = self.encoder.anchors[anchors_idx]
        self.decode = lambda x: self.encoder.decode(x, anchors_idx)

    def compute_iou(self, pred_boxes: tf.Tensor, target_boxes: tf.Tensor) -> tf.Tensor:

        pred_boxes = tf.concat([pred_boxes[..., :2] - pred_boxes[..., 2:] * 0.5,
                                pred_boxes[..., :2] + pred_boxes[..., 2:] * 0.5], axis=-1)

        target_boxes = tf.concat([target_boxes[..., :2] - target_boxes[..., 2:] * 0.5,
                                  target_boxes[..., :2] + target_boxes[..., 2:] * 0.5], axis=-1)

        boxes1_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        boxes2_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])

        left_up = tf.maximum(pred_boxes[..., :2], target_boxes[..., :2])
        right_down = tf.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area
        return iou

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:

        xt, yt, wt, ht, obj_mask, labels = tf.split(y_true, [1, 1, 1, 1, 1, self.ncls], axis=-1)
        xp, yp, wp, hp, obj_pred, logits = tf.split(y_pred, [1, 1, 1, 1, 1, self.ncls], axis=-1)

        # Compute IoU
        predicted_boxes = self.decode(y_pred)
        target_boxes = tf.reshape(y_true, [-1, *predicted_boxes.get_shape()[1:]])
        iou = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        obj_mask_bool = tf.cast(tf.reshape(obj_mask, [-1, *predicted_boxes.get_shape()[1:-1]]), tf.bool)

        def loop_body(batch_idx, iou_array):
            gt_boxes = tf.reshape(tf.boolean_mask(target_boxes[batch_idx, ..., :4], obj_mask_bool[batch_idx], axis=0), (-1, 4))
            iou_scores = compute_iou(gt_boxes, predicted_boxes[batch_idx, ..., :4])
            iou_array = iou_array.write(batch_idx, iou_scores)
            return batch_idx + 1, iou_array

        _, iou = tf.while_loop(lambda *args: args[0] < tf.shape(obj_mask)[0], loop_body, [0, iou])
        iou = iou.stack()
        iou = tf.reshape(iou, (-1, *y_pred.get_shape()[1:-1], 1))
        iou_mask = tf.cast(tf.less(iou, 0.5 * tf.ones_like(iou)), tf.float32)

        # Create grids and calculate offset targets
        x_grid = tf.tile(tf.reshape(tf.range(self.grid_size, dtype=tf.float32), [1, 1, -1, 1, 1]),
                         [1, self.grid_size, 1, 1, 1])
        y_grid = tf.tile(tf.reshape(tf.range(self.grid_size, dtype=tf.float32), [1, -1, 1, 1, 1]),
                         [1, 1, self.grid_size, 1, 1])

        offset_x = xt * self.grid_size - x_grid
        offset_y = yt * self.grid_size - y_grid
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)

        # Calculate wh targets
        anchor = tf.reshape(self.anchor, [1, 1, 1, 3, 2])
        anchor_w, anchor_h = tf.split(anchor, 2, axis=-1)
        w_target = tf.math.log(wt / anchor_w)
        h_target = tf.math.log(ht / anchor_h)
        wh_target = tf.concat([w_target, h_target], axis=-1)
        wh_target = tf.where(tf.math.is_inf(wh_target),
                             tf.zeros_like(wh_target), wh_target)

        # Promote loss for smaller boxes
        box_loss_scale = 2 - wt * ht

        # Calculate losses
        xy_loss = tf.reduce_sum(
            tf.math.squared_difference(tf.nn.sigmoid(y_pred[..., :2]), offset_xy) * obj_mask * box_loss_scale,
            axis=[1, 2, 3, 4])
        wh_loss = tf.reduce_sum(tf.math.squared_difference(y_pred[..., 2:4], wh_target) * obj_mask * box_loss_scale,
                                axis=[1, 2, 3, 4])
        cls_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
                                 axis=[1, 2, 3, 4])
        neg_obj_loss = tf.reduce_sum(
            (1 - obj_mask) * iou_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=obj_pred, labels=obj_mask),
            axis=[1, 2, 3, 4])
        pos_obj_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=obj_pred, labels=obj_mask), axis=[1, 2, 3, 4])

        return tf.reduce_mean(self.lambda_coord * xy_loss + self.lambda_coord * wh_loss + cls_loss + neg_obj_loss + pos_obj_loss)
