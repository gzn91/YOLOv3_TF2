import tensorflow as tf
import numpy as np
from utils import AnchorBoxes
from image_ops import non_max_suppression, compute_iou, read_image, draw_bbox, plot_to_image
from layers import YOLOv3Net
from losses import YoloLoss
from datetime import datetime


class YOLOv3(tf.keras.Model):

    def __init__(self, img_shape=(416,416,3), ncls=80, restore_model=True, learning_rate=0.001):
        super(YOLOv3, self).__init__()
        self._img_shape = img_shape
        self._img_size = img_shape[0]
        self._ncls = ncls

        self.anchor_manipulator = AnchorBoxes(img_size=img_shape[0], ncls=ncls)
        self.anchors = self.anchor_manipulator.anchors

        inputs = tf.keras.Input(img_shape)
        self.network = tf.keras.Model(inputs=inputs, outputs=YOLOv3Net(restore_weights=False, ncls=self._ncls, train_backbone=False)(inputs))
        if restore_model:
            self.network.load_weights('coco_init_yolov3_weights.h5')  # run layers to create this

        self.loss_52 = YoloLoss([0, 1, 2], self._img_size//8, ncls=ncls, img_shape=img_shape)
        self.loss_26 = YoloLoss([3, 4, 5], self._img_size//16, ncls=ncls, img_shape=img_shape)
        self.loss_13 = YoloLoss([6, 7, 8], self._img_size//32, ncls=ncls, img_shape=img_shape)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #  ============ Setup logger and saver =================  #
        log_dir = "logs/"

        self.summary_writer = tf.summary.create_file_writer(
            log_dir + datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.epoch = tf.Variable(1, trainable=False, dtype=tf.int64)

        checkpoint_dir = './training_checkpoints'
        self.ckpt = tf.train.Checkpoint(network=self.network,
                                        opt=self.opt,
                                        epoch=self.epoch)

        self.manger = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=3)

    def call(self, inputs, training=None, mask=None):

        return self.network(inputs, training=training)

    def loss(self, gt, yp_52, yp_26, yp_13):

        loss52 = self.loss_52(gt['y_52'], yp_52)
        loss26 = self.loss_26(gt['y_26'], yp_26)
        loss13 = self.loss_13(gt['y_13'], yp_13)

        return loss52 + loss26 + loss13 + self.losses

    @tf.function
    def train_step(self, imgs, gt):
        with tf.GradientTape() as tape:
            y_52, y_26, y_13 = self.call(imgs, training=True)
            loss = self.loss(gt, y_52, y_26, y_13)

        y_52 = self.anchor_manipulator.decode(y_52, [0, 1, 2])
        y_26 = self.anchor_manipulator.decode(y_26, [3, 4, 5])
        y_13 = self.anchor_manipulator.decode(y_13, [6, 7, 8])
        yp = tf.concat([y_52, y_26, y_13], axis=1)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.opt.apply_gradients(zip(gradients, self.trainable_variables))

        return {'Output': yp, 'loss': loss}

    @tf.function
    def _eval(self, imgs, gt):

        y_52, y_26, y_13 = self.call(imgs, training=False)
        loss = self.loss(gt, y_52, y_26, y_13)

        y_52 = self.anchor_manipulator.decode(y_52, [0, 1, 2])
        y_26 = self.anchor_manipulator.decode(y_26, [3, 4, 5])
        y_13 = self.anchor_manipulator.decode(y_13, [6, 7, 8])
        yp = tf.concat([y_52, y_26, y_13], axis=1)

        return {'Output': yp, 'loss': loss}

    def eval(self, data):

        imgs = []
        for n, (img, gt, _) in enumerate(data):
            ret = self._eval(img, gt)
            if n < 1:
                rets = {k: [] for k in ret.keys() if k != 'Output'}
                rets['iou'] = []
            [rets[k].append(v) for k, v in ret.items() if k != 'Output']
            if n % 20:
                ious = []
                yt = tf.concat([self.anchor_manipulator.decode_gt(yt) for _, yt in gt.items()], axis=1)
                for j, p in enumerate(ret['Output']):
                    scores, bboxes = non_max_suppression(p, confidence_threshold=0.2, iou_threshold=0.4)
                    scores_gt, bboxes_gt = non_max_suppression(yt[j], confidence_threshold=0.2, iou_threshold=0.4)
                    for k in bboxes.keys():
                        try:
                            iou = np.squeeze(compute_iou(bboxes[k], bboxes_gt[k]))
                        except KeyError:
                            iou = [0.0]
                        ious.append(iou)
                    fig = draw_bbox(img[j], scores, bboxes)
                    imgs.append(plot_to_image(fig))
                try:
                    ious = np.concatenate(ious, 0)
                    rets['iou'] = np.mean(ious)
                except ValueError:
                    rets['iou'] = 0.0
        for k in ret.keys():
            if k == 'Output':
                rets[k] = tf.concat(imgs, axis=0)
                continue
            rets[k] = tf.reduce_mean(rets[k])

        with self.summary_writer.as_default():
            for k, v in rets.items():
                if k == 'Output':
                    tf.summary.image('Validation Prediction', v, step=self.epoch, max_outputs=8)
                else:
                    tf.summary.scalar(k, v, self.epoch)
                    print(f'Validation {k} at epoch {self.epoch.numpy()}: {v:.3f}')
                self.summary_writer.flush()
        return ret

    def save_model(self):
        self.manger.save(self.epoch)

    def restore_model(self):
        self.ckpt.restore(self.manger.latest_checkpoint)

    def detect_and_draw(self, x):
        B = x.shape[0]
        y_52, y_26, y_13 = self.network.predict_on_batch(x)
        y_52 = self.anchor_manipulator.decode(y_52, [0, 1, 2]).reshape(B, -1, 5+self._ncls)
        y_26 = self.anchor_manipulator.decode(y_26, [3, 4, 5]).reshape(B, -1, 5+self._ncls)
        y_13 = self.anchor_manipulator.decode(y_13, [6, 7, 8]).reshape(B, -1, 5+self._ncls)

        y = np.concatenate([y_52, y_26, y_13], axis=1)

        for img, p in zip(x, y):
            scores, bboxes = non_max_suppression(p, .1)
            draw_bbox(img, scores, bboxes)

    def detect(self, x, confidence_threshold=0.5, nms_thresh=0.4):
        B = x.shape[0]
        y_52, y_26, y_13 = self.__call__(x, training=False)

        y_52 = self.anchor_manipulator.decode(y_52, [0, 1, 2])
        y_26 = self.anchor_manipulator.decode(y_26, [3, 4, 5])
        y_13 = self.anchor_manipulator.decode(y_13, [6, 7, 8])

        y = tf.concat([y_52, y_26, y_13], axis=1)

        selected_scores, selected_boxes = [], []
        for i in range(B):
            scores, bboxes = non_max_suppression(y[i], confidence_threshold, iou_threshold=nms_thresh)
            selected_scores.append(scores)
            selected_boxes.append(bboxes)

        return selected_scores, selected_boxes


from matplotlib.patches import Rectangle
import os
import matplotlib.pyplot as plt
from utils import CLASSES
# CM = plt.cm.get_cmap('Set1')
# def draw_bbox(image, preds, bboxes, dw=0, dh=0, new_sz=(416, 416)):
#     r = np.array([416,416])/np.array(new_sz)
#     print(np.int32([10*r[1], 10*r[0]]))
#     fig, ax = plt.subplots(1, figsize=np.int32([10*r[1], 10*r[0]-1]))
#     scale_h, scale_w, _ = image.shape
#     ax.imshow(image)
#     for k, v in preds.items():
#         for sc, bb in zip(preds[k], bboxes[k]):
#
#             xmin, ymin, xmax, ymax = bb
#             xmin = scale_w*(xmin - dw)*r[0]
#             xmax = scale_w*(xmax - dw)*r[0]
#             ymin = scale_h*(ymin - dh)*r[1]
#             ymax = scale_h*(ymax - dh)*r[1]
#             h = ymax - ymin
#             w = xmax - xmin
#
#             rec = Rectangle((xmin, ymin),width=w,height=h, fill=False, edgecolor=CM(k), lw=2)
#             ax.add_patch(rec)
#             ax.annotate(f'{CLASSES[k]}: {sc*100:.2f}%', (xmin, ymin), fontsize=8, color='white')
#             ax.axis('off')
#     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig('out_bb.png', bbox_inches=extent, transparent=True)
#     # plt.tight_layout()
#     # plt.show()
#     return fig


import io
import cv2



import pandas as pd
if __name__=='__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    yolo = YOLOv3()
    # yolo.train()
    # raise KeyError
    # import imageio
    # import matplotlib.pyplot as plt
    # vid = imageio.get_reader('/home/gzn/Documents/DSSD/output.mp4', 'ffmpeg')
    # figs = []
    # for num, image in enumerate(vid.iter_data()):
    #     print(num)
    #     if num == 1210:
    #         img, new_sz, dw, dh, full_res_img = read_image('', image, return_full_res_image=True)
    #         img = img / 255.
    #         scores, bboxes = yolo.detect(img[None], confidence_threshold=0.25, nms_thresh=0.5)
    #         fig = draw_bbox(full_res_img, scores[0], bboxes[0], dw, dh, new_sz)
    #         plt.show()
    #         break
    #         # figs.append(get_img_from_fig(fig))

    # imageio.mimsave('test.gif', figs)
    # imageio.mimwrite('ride.mp4', figs, fps=30)
    img, new_sz, dw, dh, full_res_img = read_image('motogp.jpg', return_full_res_image=True)
    img = img / 255.
    scores, bboxes = yolo.detect(img[None], confidence_threshold=0.15, nms_thresh=0.3)
    draw_bbox(full_res_img, scores[0], bboxes[0], dw, dh, new_sz)
    plt.show()
    # df = pd.DataFrame()
    # bboxes = pd.DataFrame([{'Class': CLASSES[k], 'Value': 1.0} for k, v in scores[0].items() for vv in v])
    # bboxes.to_csv('gc/info.csv',mode='a', header=1)
    # print(pd.DataFrame(bboxes))

