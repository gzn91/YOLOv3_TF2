import tensorflow as tf
from yolo_v3 import YOLOv3
from pathlib import PurePath
from absl import app
from absl import flags
from datetime import datetime
import os
from data_loader import DataLoader
import callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('steps_per_epoch', 1000, 'Number of steps per epoch.')
flags.DEFINE_string('model_dir', rf'saved_models', 'Directory to save model ckpts.')
flags.DEFINE_string('train_dir', '/home/gzn/Documents/MRCNN/records', 'Directory with the training data.')
flags.DEFINE_string('test_dir', 'data/test', 'Directory with the test data.')
flags.DEFINE_string('logdir', rf'logs', 'Directory with the test data.')
flags.DEFINE_integer('save_freq', 10, 'How often to save the model, specified in epochs.')
flags.DEFINE_list(
    'img_shape', [416, 416, 3], 'Size of img')
flags.DEFINE_integer(
    'mb_size', 8, 'The number of samples in each batch.')
flags.DEFINE_integer(
    'ncls', 80, 'Number of classes to use in the dataset.')
flags.DEFINE_integer(
    'max_epochs', 120, 'Number of epochs through the dataset.')
flags.DEFINE_bool(
    'restore', False, 'Restore model')
flags.DEFINE_integer(
    'initial_epoch', 0, 'The number of samples in each batch.')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def train(argv):

    loader = DataLoader(FLAGS.train_dir, n_cls=FLAGS._ncls, img_shape=FLAGS.img_shape)
    train_ds, val_ds = loader.create_dataset(batch_size=FLAGS.mb_size)
    logdir = str(PurePath(os.path.join(FLAGS.logdir, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}')))
    initial_epoch = 0

    checkpoint_path = str(PurePath(os.path.join(FLAGS.model_dir, "model-{epoch:04d}.ckpt")))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=FLAGS.save_freq)

    model = YOLOv3(img_shape=FLAGS.img_shape, ncls=FLAGS._ncls, use_spp=True, use_pretrained_weights=True)

    if FLAGS.restore:
        initial_epoch = FLAGS.initial_epoch
        path = str(PurePath(os.path.join(FLAGS.model_dir, f"model-{initial_epoch:04d}.ckpt")))

        model.load_weights(path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
    image_cb = callbacks.YOLOCallback(model.serving,
                                         val_data=val_ds,
                                         val_steps=5,
                                         logdir=logdir,
                                         encoder=loader.encoder,
                                      write_every_n_epochs=10)

    model.compile(optimizer=optimizer)
    model.fit(train_ds, steps_per_epoch=FLAGS.steps_per_epoch, validation_data=val_ds,
              validation_steps=100, epochs=FLAGS.max_epochs, callbacks=[image_cb, tb_callback, cp_callback],
              initial_epoch=initial_epoch)


app.run(train)