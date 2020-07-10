import matplotlib.pyplot as plt
from yolo_v3 import YOLOv3
from image_ops import non_max_suppression, draw_bbox, read_image
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

TARGET_SIZE = 416

model = YOLOv3(img_shape=(TARGET_SIZE, TARGET_SIZE, 3), use_spp=True, use_pretrained_weights=True)
img, new_sz, dw, dh, full_res_img = read_image('motogp.jpg', target_size=TARGET_SIZE, return_full_res_image=True)
img = img / 255.

scale = [sz / nsz for (sz, nsz) in zip(img.shape[:2], new_sz)]

preds = model.serving.predict(img[None])
scores, bboxes = non_max_suppression(preds[0])
draw_bbox(full_res_img, scores, bboxes, dw, dh, r=scale)
plt.show()
