    #coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import argparse
import shutil
from six.moves import cPickle
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

    #sess=tf.Session()
    #先加载图和参数变量
#    for op in graph.get_operations():
#      print(str(op.name))
#    var = tf.global_variables()#全部调用
#    for i in var:
#        print(i)


    input_layer= "input"
    #nput_layer = "MobilenetV2/input"
    output_layer= "MobilenetV2/Predictions/Reshape_1"
    #output_layer= "MobilenetV2/Logits/output"
    #graph = load_graph('./frozen_pb/frozen_0-0-0.pb')
    #with graph.as_default() as g:
#        image_buffer_input = g.get_tensor_by_name('input:0')
#        final_tensor = g.get_tensor_by_name('MobilenetV2/Logits/AvgPool:0')

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

    all_files_df = pd.DataFrame(columns=['image_name', 'ground_truth',"predecit_label"])

#    for label_index, label_name in enumerate(image_lists.keys()):
#      for image_index, image_name in enumerate(image_lists[label_name][category]):
#        image_name = retrain.get_image_path(
#            image_lists, label_name, image_index, image_dir, category)
##        ground_truth = np.zeros([1, class_count], dtype=np.float32)
##        ground_truth[0, label_index] = 1.0
##        ground_truths.append(ground_truth)
#        filenames.append(image_name)

#        ground_truth_argmax= np.argmax(ground_truth,axis =1)
#        ground_truth_argmax = np.squeeze(ground_truth_argmax)

        #all_files_df=all_files_df.append([{'image_name':image_name, 'ground_truth':ground_truth_argmax}],ignore_index=True)

    #all_files_df.to_csv("ground_truth1.csv")
    #print(filenames)

    filename ="/home/deepl/PHICOMM/dataset/cifar10_tf/cifar-10-batches-py/test_batch"

    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding='bytes')

    file_images = data[b'data']
    num_images = file_images.shape[0]

    file_images = file_images.reshape((num_images, 3, 32, 32))
    labels = data[b'labels']
    for i in range(num_images):
        ground_truth = np.zeros([1, class_count], dtype=np.float32)
        ground_truth[0, labels[i]] = 1.0
        ground_truths.append(ground_truth)


    if os.path.exists("./data"):
        print("data is exist, please delete it!")
        exit()
        #shutil.rmtree("./data")
    #os.makedirs("./data")


    #sio.savemat('./data/truth.mat',{"truth": ground_truths})

    cf = 0.875
    predictions = []
    i = 0
    start = time.time()



    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    record_iterator = tf.python_io.tf_record_iterator(path='/home/deepl/PHICOMM/dataset/cifar10_tf/cifar10_test.tfrecord')
    c =0
#    for string_iterator in record_iterator:
#        c += 1
#        example = tf.train.Example()
#        example.ParseFromString(string_iterator)
#        height = example.features.feature['image/height'].int64_list.value[0]
#        width = example.features.feature['image/width'].int64_list.value[0]
#        png_string = example.features.feature['image/encoded'].bytes_list.value[0]
#        label = example.features.feature['image/class/label'].int64_list.value[0]
    example = tf.train.Example()


    with tf.Session(config=config) as sess:
    #with tf.Session(graph=graph) as sess:
        network_fn = nets_factory.get_network_fn(
            "mobilenet_v2",
            num_classes=10,
            is_training=False)
        image_size = 224
        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=[128, image_size,
                                            image_size, 3])
        logits, _ = network_fn(placeholder)
        graph = tf.get_default_graph()
        saver = tf.train.Saver()
        #raph_def = graph.as_graph_def()
        #aver = tf.train.import_meta_graph(graph_def)

        #saver = tf.train.import_meta_graph('./mobilenetv2_on_cifar10_check_point/0/model_0/model.ckpt-20000.meta')
        saver.restore(sess,'./mobilenetv2_on_cifar10_check_point/0/model_0/model.ckpt-20000')
        # Initalize the variables
        #sess.run(tf.global_variables_initializer())

        output_operation = graph.get_operation_by_name(output_layer);
        input_operation = graph.get_operation_by_name(input_layer);
        read_tensor_from_jpg_image_file_op = read_tensor_from_jpg_image_file(
                           input_height=224,
                           input_width=224)
        #file_num = len(filenames)
        file_num = 10000
        batch_size = 128
        count = file_num // batch_size
        print("need %d batch, every batch is %d"%(count,batch_size))

        image_placeholder = tf.placeholder(dtype=tf.uint8)
        #decoded_img = tf.image.decode_png(image_placeholder, channels=3)
        image = tf.transpose(image_placeholder,[1, 2, 0])
        normalized = preprocess_for_eval(image, 224, 224)
        for i in range(count):
            print("this is %d batch"%i)
            print("jpg order get from %d to %d"%(batch_size*i,batch_size*i+batch_size-1))
            for j in range(batch_size):
                if j == 0:
                  # t_batch = sess.run(read_tensor_from_jpg_image_file_op,feed_dict={"fnamejpg:0": filenames[batch_size*i+j]})
                    t_batch = sess.run(normalized,feed_dict={image_placeholder: file_images[batch_size*i+j]})
                    t_batch = np.expand_dims(t_batch,axis = 0)

                else:
                    t = sess.run(normalized,feed_dict={image_placeholder: file_images[batch_size*i+j]})
                    t = np.expand_dims(t,axis = 0)
                    t_batch = np.concatenate((t_batch,t), axis=0)
            print(t_batch.shape)
#            feed_dict={image_buffer_input: t}
#            ft = final_tensor.eval(feed_dict, sess)
#           pre = sess.run(output_operation.outputs[0],
#                      {input_operation.outputs[0]: t_batch})
            pre = sess.run(logits,
                       {input_operation.outputs[0]: t_batch})

            print(pre.shape)

            for k in range(batch_size):
                predictions.append(pre[k,:])

        if file_num % batch_size !=0:
            print("this is %d batch"%count)
            print("jpg order get from %d to %d"%(batch_size*count,file_num-1))
            extra_num = file_num - count*batch_size
            for j in range(extra_num):
                if j == 0:
                    t_batch = sess.run(normalized,feed_dict={image_placeholder: file_images[batch_size*count+j]})
                    t_batch = np.expand_dims(t_batch,axis = 0)
                else:
                    t = sess.run(normalized,feed_dict={image_placeholder: file_images[batch_size*count+j]})
                    t = np.expand_dims(t,axis = 0)
                    t_batch = np.concatenate((t_batch,t), axis=0)
            print(t_batch.shape)

            for j in range(extra_num,batch_size):
                t_batch = np.concatenate((t_batch,t), axis=0)
            print(t_batch.shape)

#           pre = sess.run(output_operation.outputs[0],
#                        {input_operation.outputs[0]: t_batch})
            pre = sess.run(logits,
                       {input_operation.outputs[0]: t_batch})
            for k in range(extra_num):
                predictions.append(pre[k,:])

            #i = i + 1
            #print(i)
    predictions = np.array(predictions)
    #print(predictions.shape)
    #predictions = np.squeeze(predictions)
    ground_truths = np.array(ground_truths)
    ground_truths = np.squeeze(ground_truths)

    print(predictions.shape)
    print(ground_truths.shape)

    with tf.Session(config=config) as sess:
   # with tf.Session(graph=graph) as sess:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, 10], name='GroundTruthInput')
        fts = tf.placeholder(tf.float32, [None, 10], name='fts')
        accuracy, _ = retrain.add_evaluation_step(fts, ground_truth_input)
        feed_dict={fts: predictions, ground_truth_input: ground_truths}
        #accuracies.append(accuracy.eval(feed_dict, sess))
        ret = accuracy.eval(feed_dict, sess)

#    for index, row in all_files_df.iterrows():
#        row['predecit_label'] = np.squeeze(np.argmax(predictions[index,:],axis=0))

#    all_files_df.to_csv("ground_truth.csv")
    print('Ensemble Accuracy: %g' % ret)

    stop = time.time()
    #print(str((stop-start)/len(ftg))+' seconds.')
    #sio.savemat('./data/feature.mat',{"feature": ftg})
    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    extract()






