#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import argparse
import shutil

import numpy as np
import PIL.Image as Image
import tensorflow as tf
import pandas as pd
import retrain as retrain
from count_ops import load_graph

import time

import scipy.io as sio

sys.path.append("/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim")
from nets import nets_factory
from datasets import dataset_factory
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './mobilenetv2_on_cifar10_check_point/0/model_0/model.ckpt-20000',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')


tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', "/home/deepl/PHICOMM/dataset/cifar10_tf", 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
  if central_fraction:
    image = tf.image.central_crop(image, central_fraction=central_fraction)

  if height and width:
      # Resize the image to the specified height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
    image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

def read_tensor_from_jpg_image_file(input_height=299, input_width=299,
        input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    # [NEW] make file_name as a placeholder.
    file_name_placeholder = tf.placeholder("string", name="fnamejpg")

    file_reader = tf.read_file(file_name_placeholder, input_name)
    #  if file_name.endswith(".png"):
    #    image_reader = tf.image.decode_png(file_reader, channels = 3,
    #                                       name='png_reader')
    #  elif file_name.endswith(".gif"):
    #    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
    #                                                  name='gif_reader'))
    #  elif file_name.endswith(".bmp"):
    #    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    #  else:
    #    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
    #                                        name='jpeg_reader')
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
    normalized = preprocess_for_eval(image_reader, input_height, input_width)
    #sess = tf.Session()
    #result = sess.run(normalized)
    #return result
    return normalized

def read_and_decode(filename_queue,image_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
              'image/encoded': tf.FixedLenFeature([], tf.string),
              'image/class/label': tf.FixedLenFeature([], tf.int64),
          })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    decoded_img = tf.image.decode_png(features['image/encoded'], channels=3)
    print(decoded_img.shape)
   # decoded_img = tf.decode_raw(features['image/encoded'],out_type=tf.uint8)
    decoded_img= tf.reshape(decoded_img,shape=[32,32,3])
    #decoded_img.set_shape([224,224,3])
    #image.set_shape([mnist.IMAGE_PIXELS])
    #print("image:",image)

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    image = preprocess_for_eval(decoded_img, image_size, image_size)
    label = tf.cast(features['image/class/label'], tf.int32)
    print("label:",label)

    return image, label



def extract():

    input_layer= "input"
    #nput_layer = "MobilenetV2/input"
    output_layer= "MobilenetV2/Predictions/Reshape_1"

    total_start = time.time()
#    if os.path.exists("./data"):
#        print("data is exist, please delete it!")
#        exit()
        #shutil.rmtree("./data")
    #os.makedirs("./data")


    #sio.savemat('./data/truth.mat',{"truth": ground_truths})

    cf = 0.875
    predictions = []
    ground_truths = []
    i = 0
    start = time.time()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_epochs=1,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        image = preprocess_for_eval(image, FLAGS.eval_image_size, FLAGS.eval_image_size)

#        filename_queue = tf.train.string_input_producer(
#            [FLAGS.dataset_dir], num_epochs=1)
#        image, label = read_and_decode(filename_queue,FLAGS.eval_image_size)
#        print(image.shape)
#        print(image)
#        print(label.shape)

        images_batch, labels_batch = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size,
            allow_smaller_final_batch=True)

        labels_batch_one_hot = tf.one_hot(labels_batch,dataset.num_classes)


        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=[None, FLAGS.eval_image_size,
                                            FLAGS.eval_image_size, 3])
        logits, _ = network_fn(placeholder)
        graph = tf.get_default_graph()
        saver = tf.train.Saver()

        output_operation = graph.get_operation_by_name(output_layer);
        input_operation = graph.get_operation_by_name(input_layer);

        batch_size = FLAGS.batch_size
        print("every batch is %d"%(batch_size))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess,FLAGS.checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        count = 0
        try:
            while not coord.should_stop():
                image_batch_v, label_batch_v = sess.run([images_batch, labels_batch_one_hot])
                #print(image_batch_v.shape, label_batch_v.shape)
                print("this is %d batch"%count)

                ground_truths.extend(label_batch_v)
                count += 1
                pre = sess.run(logits,
                       {input_operation.outputs[0]: image_batch_v})

                predictions.extend(pre)

        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)

            #i = i + 1
            #print(i)
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)


    print(predictions.shape)
    print(ground_truths.shape)

    with tf.Session(config=config) as sess:
   # with tf.Session(graph=graph) as sess:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, 10], name='GroundTruthInput')
        predicts = tf.placeholder(tf.float32, [None, 10], name='predicts')
        accuracy, _ = retrain.add_evaluation_step(predicts, ground_truth_input)
        feed_dict={predicts: predictions, ground_truth_input: ground_truths}
        #accuracies.append(accuracy.eval(feed_dict, sess))
        ret = accuracy.eval(feed_dict, sess)

    print('Ensemble Accuracy: %g' % ret)

    stop = time.time()
    #print(str((stop-start)/len(ftg))+' seconds.')
    #sio.savemat('./data/feature.mat',{"feature": ftg})
    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    extract()






