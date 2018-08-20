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
import xlsxwriter

sys.path.append("/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim")
from nets import nets_factory
from datasets import dataset_factory
#workspace="/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX"
workspace = "."


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
    'dataset_dir', "/home/deepl/PHICOMM/dataset/cifar10_tf/cifar10_test.tfrecord", 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to evaluate.')


tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

FLAGS = tf.app.flags.FLAGS









def extract():

    prediction_path=workspace+'/prediction'

    if os.path.exists(prediction_path):
        print("%s is exist, will rewrite it!"%prediction_path)
        #exit()
        #shutil.rmtree(prediction_path)
    else:
        os.makedirs(prediction_path)

    all_checkpoints = glob(os.path.join(FLAGS.checkpoint_path, "*.data*"))
    #print(all_checkpoints)

    input_layer= "MobilenetV2/Logits/AvgPool"
    output_layer= "MobilenetV2/Predictions/Reshape_1"

    feature = sio.loadmat(workspace+'/data/feature.mat')
    truth = sio.loadmat(workspace+'/data/truth.mat')
    ftg = feature['feature']
    ground_truths = truth['truth']

    total_start = time.time()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    workbook = xlsxwriter.Workbook("accuracies3.xlsx")
    worksheet = workbook.add_worksheet()


    #print(ground_truths.shape)

    with tf.Session(config=config) as sess:
   # with tf.Session(graph=graph) as sess:
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=[None, FLAGS.eval_image_size,
                                            FLAGS.eval_image_size, 3])

        logits, _ = network_fn(placeholder)
        graph = tf.get_default_graph()
        saver = tf.train.Saver()
        output_operation = graph.get_operation_by_name(output_layer);
        input_operation = graph.get_operation_by_name(input_layer);

        ground_truth_input = tf.placeholder(
                    tf.float32, [None, dataset.num_classes], name='GroundTruthInput')
        predicts = tf.placeholder(tf.float32, [None, dataset.num_classes], name='predicts')
        accuracy, _ = retrain.add_evaluation_step(predicts, ground_truth_input)

        for i,checkpoint in enumerate(all_checkpoints):
            checkpoint_prefix = checkpoint.replace('.data-00000-of-00001', '')
            saver.restore(sess,checkpoint_prefix)

            predictions = sess.run(output_operation.outputs[0],
                 {input_operation.outputs[0]: ftg})

            #print(predictions.shape)

            feed_dict={predicts: predictions, ground_truth_input: ground_truths}
            #accuracies.append(accuracy.eval(feed_dict, sess))
            ret = accuracy.eval(feed_dict, sess)

            _,fname=os.path.split(checkpoint_prefix)
            prediction_name = fname.replace("model.ckpt-","")

            worksheet.write(i, 0, fname)
            worksheet.write(i, 1, ret)
            print('checkpoint: %s, Ensemble Accuracy: %g' % (checkpoint,ret))
            sio.savemat(workspace+'/prediction/prediction_'+prediction_name+'.mat',{"prediction": predictions})

    stop = time.time()
    #print(str((stop-start)/len(ftg))+' seconds.')
    #sio.savemat('./data/feature.mat',{"feature": ftg})
    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    extract()






