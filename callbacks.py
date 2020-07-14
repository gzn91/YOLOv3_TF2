import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import io
import image_ops
from anchors import CLASSES, AnchorBoxes


class ImageCallback(tf.keras.callbacks.Callback):

    def __init__(self, model: tf.keras.Model, val_data: tf.data.Dataset, val_steps: int, logdir: str, encoder: AnchorBoxes):
        super(ImageCallback, self).__init__()
        self._model = model
        self.val_data = val_data
        self.val_steps = val_steps
        self.encoder = encoder
        self.filewriter = tf.summary.create_file_writer(logdir)

    def plot_to_image(self, figure: plt.Figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.convert_image_dtype(tf.image.decode_png(buf.getvalue(), channels=4), tf.uint8, saturate=True)
        buf.close()
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def draw_bboxes_gt(self, img: tf.Tensor, bboxes: tf.Tensor, classes: tf.Tensor) -> plt.Figure:
        CM = plt.cm.get_cmap('Set1')
        fig, ax = plt.subplots(figsize=(8,8))
        scale_h, scale_w, _ = img.shape
        ax.imshow(img)
        for bbox, cls in zip(bboxes, classes):
            xmin, ymin, xmax, ymax = tf.split(bbox, 4, axis=-1)
            xmin *= scale_w
            xmax *= scale_w
            ymin *= scale_h
            ymax *= scale_h
            h = ymax - ymin
            w = xmax - xmin
            rec = Rectangle((xmin, ymin), width=w, height=h, fill=False, edgecolor=CM(cls), lw=4)
            ax.add_patch(rec)
            ax.annotate(f'{CLASSES[cls]}', (xmin, ymin), fontsize=8, color='white')
            ax.axis('off')
        return fig

    @tf.function
    def predict(self, x: tf.Tensor) -> tf.Tensor:

        return self._model(x, training=False)


class YOLOCallback(ImageCallback):

    def __init__(self, model: tf.keras.Model, val_data: tf.data.Dataset,
                 val_steps: int, logdir: str, encoder: AnchorBoxes, write_every_n_epochs: int = 1):
        super(YOLOCallback, self).__init__(model, val_data, val_steps, logdir, encoder)
        self.write_every_n_epochs = write_every_n_epochs
        self.filewriter = tf.summary.create_file_writer(logdir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.write_every_n_epochs == 0:
            with self.filewriter.as_default():
                pred_boxes_ = []
                gt_boxes_ = []
                ious = []
                for j, (input, _) in enumerate(self.val_data):
                    imgs = input['input']
                    gt_boxes = input['gt_boxes']
                    gt_cls = input['gt_cls']
                    y_pred = self.predict(input)
                    for i, img in enumerate(imgs):
                        detections = y_pred[i]

                        scores_, bboxes_ = image_ops.non_max_suppression(detections,
                                                                        iou_threshold=0.4,
                                                                        confidence_threshold=0.4,
                                                                        max_nr_boxes_per_cls=256)

                        fig = image_ops.draw_bbox(img, scores_, bboxes_)
                        pred_boxes_.append(self.plot_to_image(fig))

                        bbox = gt_boxes[i]
                        cls = gt_cls[i]
                        idx_to_keep = tf.reduce_sum(tf.abs(bbox), -1) > 0.001
                        bbox = bbox[idx_to_keep]
                        cls = cls[idx_to_keep]
                        fig = self.draw_bboxes_gt(img, bbox, cls)
                        gt_boxes_.append(self.plot_to_image(fig))

                        # RPN IOU
                        for cls_id in np.unique(cls):
                            try:
                                cls_bbox = bbox[cls == cls_id]
                                iou = image_ops.compute_iou(bboxes_[cls_id], cls_bbox)
                            except KeyError:
                                iou = np.zeros(cls_bbox.shape[0])
                            ious.append(iou)

                    if j > self.val_steps:
                        break

                pred_boxes_ = tf.concat(pred_boxes_, 0)
                gt_boxes_ = tf.concat(gt_boxes_, 0)
                ious = tf.concat(ious, 0)
                avg_iou = tf.reduce_mean(ious)

                tf.summary.scalar('Validation IoU', avg_iou, step=epoch)
                tf.summary.image(f'Predicted Boxes', pred_boxes_, step=epoch, max_outputs=12)
                tf.summary.image(f'GT Boxes', gt_boxes_, step=epoch, max_outputs=12)
