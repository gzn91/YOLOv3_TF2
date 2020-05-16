import tensorflow as tf
import numpy as np
from yolo_v3 import YOLOv3
from absl import app
from absl import flags
from datetime import datetime
import os
from data_loader import DataLoader
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data/', 'Directory with the training data.')
flags.DEFINE_string('test_dir', 'data/test', 'Directory with the test data.')
flags.DEFINE_list(
    'img_shape', [416, 416, 3], 'Size of img')
flags.DEFINE_integer(
    'mb_size', 4, 'The number of samples in each batch.')
flags.DEFINE_integer(
    'ncls', 80, 'Number of classes to use in the dataset.')
flags.DEFINE_integer(
    'nepochs', 100000, 'Number of epochs through the dataset.')
flags.DEFINE_bool(
    'restore', False, 'Restore model')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.backend.set_floatx('float16')

def train(argv):

    # train_ds, val_ds = loader.generate_dataset()
    model = YOLOv3(img_shape=FLAGS.img_shape, ncls=FLAGS.ncls, restore_model=True, learning_rate=FLAGS.learning_rate)
    loader = DataLoader(FLAGS.train_dir, n_cls=FLAGS.ncls, img_shape=FLAGS.img_shape)
    train_ds, val_ds = loader.create_dataset(batch_size=FLAGS.mb_size)

    N = len(list(train_ds)) - 1 // FLAGS.mb_size

    if FLAGS.restore:
        model.restore_model()

    starting_epoch = int(model.epoch)

    for epoch in range(starting_epoch, FLAGS.nepochs):
        start_ = time.time()

        # Train
        mb_time = 1.0
        for n, (imgs, gt, filenames) in enumerate(train_ds):
            start = time.time()
            string = '\r' + f'[{"="*n}>{"."*(N-n-1)}] - ETA: {((N-n-1)*mb_time):.3f}'
            ret = model.train_step(imgs, gt)
            print(string + '\t' + ', '.join([f'{k}: {np.mean(v):.3f}' for k, v in ret.items() if k != 'Output']), end='')
            end = time.time()
            mb_time = end-start
        print()
        print()

        ret = model.eval(val_ds)
        model.epoch.assign_add(1)
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.save_model()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start_))


app.run(train)