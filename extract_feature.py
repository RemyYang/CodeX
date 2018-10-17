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
from glob import glob
import time

import scipy.io as sio

from nets import nets_factory
from datasets import dataset_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim


tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './renamed_check_point',
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

tf.app.flags.DEFINE_string(
    'input_layer', "input", 'input_layer')

tf.app.flags.DEFINE_string(
    'output_layer', "MobilenetV2/Logits/AvgPool", 'output_layer')

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

    total_start = time.time()
    if os.path.exists("./data"):
        print("data is exist, will rewrite it!")
        #exit()
        #shutil.rmtree("./data")
    else:
        os.makedirs("./data")


    predictions = []
    ground_truths = []
    features=[]
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

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)


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

        output_operation = graph.get_operation_by_name(FLAGS.output_layer);
        input_operation = graph.get_operation_by_name(FLAGS.input_layer);

        batch_size = FLAGS.batch_size
        print("every batch is %d"%(batch_size))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess,FLAGS.checkpoint_path+"/model.ckpt-0-0-0")

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
                feature = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: image_batch_v})


                features.extend(feature)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)
            #i = i + 1
            #print(i)
    features = np.array(features)
    ground_truths = np.array(ground_truths)



    print(features.shape)
    print(ground_truths.shape)#
    truth_mat_name='./data/'+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+'truth.mat'
    Feature_mat_name='./data/'+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+'feature.mat'
    sio.savemat(truth_mat_name,{"truth": ground_truths})
    sio.savemat(Feature_mat_name,{"feature": features})


########test accuracy
#    with tf.Session(graph=graph) as sess:
#        output_operation = graph.get_operation_by_name("MobilenetV2/Predictions/Reshape_1");
#        input_operation = graph.get_operation_by_name(FLAGS.output_layer);

#        ground_truth_input = tf.placeholder(
#                    tf.float32, [None, dataset.num_classes], name='GroundTruthInput')
#        predicts = tf.placeholder(tf.float32, [None, dataset.num_classes], name='predicts')
#        accuracy, _ = retrain.add_evaluation_step(predicts, ground_truth_input)

#        #saver.restore(sess,"/home/deepl/Project/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim/mobilenetv2_quantize_checkpoint/base/model.ckpt-40000")
#        saver.restore(sess,FLAGS.checkpoint_path+"/model.ckpt-best-0")
#        predictions = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: features})
#        feed_dict={predicts: predictions, ground_truth_input: ground_truths}
#        #accuracies.append(accuracy.eval(feed_dict, sess))
#        ret = accuracy.eval(feed_dict, sess)
#    print('Ensemble Accuracy: %g' % ret)


    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    extract()






