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


def extract():

    input_layer= "input"
    output_layer= "MobilenetV2/Logits/AvgPool"
    graph = load_graph('./base/base.pb')
    #with graph.as_default() as g:
#        image_buffer_input = g.get_tensor_by_name('input:0')
#        final_tensor = g.get_tensor_by_name('MobilenetV2/Logits/AvgPool:0')
    input_operation = graph.get_operation_by_name(input_layer);
    output_operation = graph.get_operation_by_name(output_layer);

    #image_dir = '/home/xxxx/PHICOMM/ai-share/dataset/imagenet/raw-data/imagenet-data/validation'
    image_dir="/home/deepl/PHICOMM/dataset/cifar10_tf/cifar-10/test"
    testing_percentage = 100
    validation_percentage = 0
    category='testing'

    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage,
        validation_percentage)
    class_count = len(image_lists.keys())
    print(class_count)

    total_start = time.time()

    ground_truths = []
    filenames = []

    #all_files_df = pd.DataFrame(columns=['image_name', 'ground_truth'])

    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(image_lists[label_name][category]):
        image_name = retrain.get_image_path(
            image_lists, label_name, image_index, image_dir, category)
        ground_truth = np.zeros([1, class_count], dtype=np.float32)
        ground_truth[0, label_index] = 1.0
        ground_truths.append(ground_truth)
        filenames.append(image_name)

        #all_files_df=all_files_df.append([{'image_name':image_name, 'ground_truth':ground_truth}],ignore_index=True)

    #all_files_df.to_csv("ground_truth.csv")

    if os.path.exists("./data"):
        print("data is exist, please delete it!")
        exit()
        #shutil.rmtree("./data")
    os.makedirs("./data")


    sio.savemat('./data/truth.mat',{"truth": ground_truths})

    cf = 0.875
    ftg = []
    i = 0
    start = time.time()
    with tf.Session(graph=graph) as sess:
        read_tensor_from_jpg_image_file_op = read_tensor_from_jpg_image_file(
                           input_height=224,
                           input_width=224)
        for filename in filenames:
            t = sess.run(read_tensor_from_jpg_image_file_op,feed_dict={"fnamejpg:0": filename})
            t = np.expand_dims(t,axis = 0)
            #print(t.shape)
#            feed_dict={image_buffer_input: t}
#            ft = final_tensor.eval(feed_dict, sess)
            ft = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: t})
            ftg.append(ft)
            #i = i + 1
            #print(i)
    stop = time.time()
    print(str((stop-start)/len(ftg))+' seconds.')
    sio.savemat('./data/feature.mat',{"feature": ftg})
    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    extract()






