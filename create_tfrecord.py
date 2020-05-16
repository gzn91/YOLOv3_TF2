import numpy as np
import cv2
import os
import glob
import PIL
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from lxml import etree
from image_ops import read_image
from absl import app
from absl import flags
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None, 'Path to images')
flags.DEFINE_string('output_dir', None, 'Path to save the tfrecords')

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


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       image_subdirectory):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    # img, new_sz, dw, dh, img_hr = read_image(full_path, return_full_res_image=True)
    # r = np.array([416,416])/np.array(new_sz)

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    img = tf.image.decode_jpeg(encoded_jpg)
    # img_crop_pil = PIL.Image.fromarray(img)
    # with io.BytesIO() as byte_io:
    #     img_crop_pil.save(byte_io, format="jpeg")
    #     encoded_jpg = byte_io.getvalue()

    height, width = img.shape[:-1]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    # ax.imshow(img)
    if 'object' in data:
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin']) / width
            ymin = float(obj['bndbox']['ymin']) / height
            xmax = float(obj['bndbox']['xmax']) / width
            ymax = float(obj['bndbox']['ymax']) / height

            # xmin = float(obj['bndbox']['xmin']) / w_
            # ymin = float(obj['bndbox']['ymin']) / h_
            # xmax = float(obj['bndbox']['xmax']) / w_
            # ymax = float(obj['bndbox']['ymax']) / h_
            # xmin = new_sz[0]*xmin/width + dw
            # ymin = new_sz[1]*ymin/height + dh
            # xmax = new_sz[0]*xmax/width + dw
            # ymax = new_sz[1]*ymax/height + dh

            # xmin = scale_w*xmin
            # xmax = scale_w*xmax
            # ymin = scale_h*ymin
            # ymax = scale_h*ymax
            # h = ymax - ymin
            # w = xmax - xmin
            #
            # rec = Rectangle((xmin, ymin), width=w, height=h, edgecolor='g', fill=False, lw=2)
            # ax.add_patch(rec)
            # ax.axis('off')

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
        plt.show()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(height),
        'width': int64_feature(width),
        'filename': bytes_feature(
            data['filename'].encode('utf8')),
        'image': bytes_feature(encoded_jpg),
        'xmin': float_list_feature(xmins),
        'xmax': float_list_feature(xmaxs),
        'ymin': float_list_feature(ymins),
        'ymax': float_list_feature(ymaxs),
        'class': bytes_list_feature(classes_text),
        'label': int64_list_feature(classes),
    }))
    return example


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(argv):
    if not FLAGS.input_dir:
        return f'Path to data is not defined, input dir = {FLAGS.input_dir}'

    if not FLAGS.output_dir:
        return f'Output dir is not valid, Output dir = {FLAGS.output_dir}'

    data_dir = FLAGS.input_dir
    label_map_dict = {'person': 0,
                      'bicycle': 1,
                      'car': 2,
                      'motorbike': 3,
                      'aeroplane': 4,
                      'bus': 5,
                      'train': 6,
                      'truck': 7,
                      'boat': 8,
                      'traffic light': 9}

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    annotations_dir = os.path.join(data_dir, 'annotations')
    img_dir = os.path.join(data_dir, 'imgs')
    img_list = sorted(os.listdir(img_dir))

    N = len(img_list)
    N_train = int(0.8*N)

    train_set = img_list[:N_train]
    validation_set = img_list[N_train:]
    for record_name, dataset in zip(['training', 'validation'], [train_set, validation_set]):

        output_path = os.path.join(FLAGS.output_dir, record_name+'.tfrecord')
        with tf.io.TFRecordWriter(output_path) as writer:

            logging.info('Reading dataset.')

            for idx, example in enumerate(dataset):
                if idx % 100 == 0:
                    logging.info('On image %d of %d' % (idx+1, len(img_list)))
                example = os.path.splitext(example)[0]
                path = os.path.join(annotations_dir, example + '.xml')
                with tf.io.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = recursive_parse_xml_to_dict(xml)['annotation']

                tf_example = dict_to_tf_example(data, data_dir, label_map_dict, 'imgs')
                writer.write(tf_example.SerializeToString())

        with open(os.path.join(FLAGS.output_dir, record_name+'.json'), 'w') as f:
            json.dump({record_name: dataset}, f, indent='\t')
    print('Done!')


app.run(main)
