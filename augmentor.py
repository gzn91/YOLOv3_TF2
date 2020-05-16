import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import matplotlib.pyplot as plt

class Augmentor(object):

    def __init__(self):
        self.seq = iaa.Sequential([iaa.Affine(
                                       translate_percent=(-0.15, 0.15),
                                       scale=(0.8, 1.2),
                                       rotate=(-3., 3.),
                                       shear=(-1., 1.)),
                                   iaa.Fliplr(0.5)])

    def augment(self, img, bboxes, labels):
        h, w, c = img.shape
        bboxes[:, 3] = (bboxes[:, 1] + bboxes[:, 3]) * h
        bboxes[:, 2] = (bboxes[:, 0] + bboxes[:, 2]) * w
        bboxes[:, 1] *= h
        bboxes[:, 0] *= w

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=label)
            for label, bbox in zip(labels, bboxes)], shape=(h, w, c))

        # img1 = bbs.draw_on_image(np.uint8(img[..., 2:5] * 255))
        # plt.imshow(img1[..., 1], cmap='gray')
        # plt.savefig('preaug.png')

        # Augment BBs and images.
        img_aug, bbs_aug = self.seq(image=img, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        labels_aug = np.array([bb.label for bb in bbs_aug.bounding_boxes])

        # img1 = bbs_aug.draw_on_image(np.uint8(img_aug[..., 2:5] * 255))
        # plt.imshow(img1[..., 1], cmap='gray')
        # plt.savefig('postaug.png')
        # raise KeyError

        if len(labels_aug) < 1:  # revert to non aug
            boxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0],
                              bboxes[:, 3] - bboxes[:, 1]], axis=-1) / img.shape[0]
            return img, boxes, labels
        else:
            bboxes = bbs_aug.to_xyxy_array()
            boxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0],
                              bboxes[:, 3] - bboxes[:, 1]], axis=-1) / img.shape[0]

            return img_aug, boxes, labels_aug





