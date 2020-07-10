import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np


class Augmentor(object):

    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                                     iaa.AddToHueAndSaturation((-40, 40)),
                                     iaa.AddToBrightness((-40, 40))
                                    ])),
            iaa.Affine(
                       translate_percent=(-0.15, 0.15),
                       scale=(0.8, 1.2),
                       rotate=(-5., 5.),
                       shear=(-2., 2.)),
            iaa.Sometimes(0.1, iaa.Dropout(0.10))
                                    ])

    def augment(self, img, bboxes, labels):
        size, _, c = img.shape

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=label)
            for label, bbox in zip(labels, bboxes)], shape=(size, size, c))

        img_aug, bbs_aug = self.seq(image=img, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        labels_aug = np.array([bb.label for bb in bbs_aug.bounding_boxes])

        bboxes = bbs_aug.to_xyxy_array()

        return img_aug.astype(np.float32), bboxes, np.int64(labels_aug)






