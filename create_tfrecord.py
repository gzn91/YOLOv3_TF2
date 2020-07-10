import numpy as np
import os
import PIL
import io
import tensorflow as tf
import json
from absl import app
from absl import flags
from pycocotools import mask
import logging
from anchors import CLASSES

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'data/train2017', 'Path to images')
flags.DEFINE_string('output_dir', 'records', 'Path to save the tfrecords')
flags.DEFINE_string('annotation_dir', 'data/annotations/instances_train2017.json', 'Path to annotaions')
flags.DEFINE_list('num_shards', [100, 10], 'shards in record')
flags.DEFINE_bool('include_masks', True, 'Write masks to record')

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.
    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.
    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.
    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = tf.image.decode_jpeg(encoded_jpg)
    if image.shape[-1] < 3:
        return False, None

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        category_id = CLASSES.index(category_index[category_id]['name'])
        category_ids.append(category_id)
        area.append(object_annotations['area'])

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
    if len(category_ids) == sum(is_crowd):
        print('no valid annotations')
        return False, None
    feature_dict = {
        'height':
            int64_feature(image_height),
        'width':
            int64_feature(image_width),
        'filename':
            bytes_feature(filename.encode('utf8')),
        'image':
            bytes_feature(encoded_jpg),
        'xmin':
            float_list_feature(xmin),
        'xmax':
            float_list_feature(xmax),
        'ymin':
            float_list_feature(ymin),
        'ymax':
            float_list_feature(ymax),
        'text':
            bytes_list_feature(category_names),
        'label':
            int64_list_feature(category_ids),
        'is_crowd':
            int64_list_feature(is_crowd),
        'area':
            float_list_feature(area),
    }
    if include_masks:
        feature_dict['mask'] = (
            bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return True, example


def main(argv):
    if not FLAGS.input_dir:
        return f'Path to data is not defined, input dir = {FLAGS.input_dir}'

    if not FLAGS.output_dir:
        return f'Output dir is not valid, Output dir = {FLAGS.output_dir}'

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    with open(FLAGS.annotation_dir, 'r') as f:
        groundtruth_data = json.load(f)

    images = groundtruth_data['images']
    category_index = create_category_index(groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
        for annotation in groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
                annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            missing_annotation_count += 1
            annotations_index[image_id] = []

    N = len(images)
    train_set = images[:int(0.9*N)]
    validation_set = images[int(0.9*N):]
    for (record_name, dataset, num_shards) in zip(['training', 'validation'], [train_set, validation_set], FLAGS.num_shards):

        writers = [tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir, record_name+f'-{(i+1):02d}-{num_shards}.tfrecord')) for i in range(num_shards)]

        for idx, image in enumerate(dataset):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            valid, tf_example = create_tf_example(
                image, annotations_list, FLAGS.input_dir, category_index, FLAGS.include_masks)
            if not valid:
                continue
            shard_idx = idx % num_shards
            writers[shard_idx].write(tf_example.SerializeToString())

    print('Done!')


app.run(main)