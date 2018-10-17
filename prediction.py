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

import retrain as retrain
from count_ops import load_graph
from glob import glob
import time

import scipy.io as sio
import xlsxwriter

from nets import nets_factory
from datasets import dataset_factory
import pandas as pd
import json
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

tf.app.flags.DEFINE_string(
    'input_layer', "MobilenetV2/Logits/AvgPool", 'input_layer')

tf.app.flags.DEFINE_string(
    'output_layer', "MobilenetV2/Predictions/Reshape_1", 'output_layer')

FLAGS = tf.app.flags.FLAGS




def saveDataFramToFile(df,fileName):
    data = {}
    datalist= df.to_dict('record')
    data["accuracy_list"]=datalist
    '''
    if df.shape[0] > 0:
        data_total['msg']="ok"
        data_total['code']="0"
    else:
        data_total['msg']="error"
        data_total['code']="-1"
    '''
    jsObj = json.dumps(data,ensure_ascii=False)
    f = file(fileName,'w')
    f.write(jsObj)
    f.close()
    return data




def extract():

    prediction_path=workspace+'/prediction/'+FLAGS.model_name+'_'+FLAGS.dataset_name

    if os.path.exists(prediction_path):
        print("%s is exist, will rewrite it!"%prediction_path)
        #exit()
        #shutil.rmtree(prediction_path)
    else:
        os.makedirs(prediction_path)

    all_checkpoints = glob(os.path.join(FLAGS.checkpoint_path, "*.data*"))
    old_checkpoints = glob(os.path.join(FLAGS.checkpoint_path, "*.ckpt"))
    all_checkpoints = all_checkpoints + old_checkpoints
    all_checkpoints.sort()
    print(all_checkpoints)
s

    truth_mat_name=workspace+'/data/'+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+'truth.mat'
    Feature_mat_name=workspace+'/data/'+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+'feature.mat'

    feature = sio.loadmat(Feature_mat_name)
    truth = sio.loadmat(truth_mat_name)
    ftg = feature['feature']
    ground_truths = truth['truth']

    total_start = time.time()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    workbook = xlsxwriter.Workbook("accuracies3.xlsx")
    worksheet = workbook.add_worksheet()

    all_files_df = pd.DataFrame(columns=['checkpoint_name', 'accuracy'])



    #print(ground_truths.shape)

    with tf.Session(config=config) as sess:
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
        output_operation = graph.get_operation_by_name(FLAGS.output_layer);
        input_operation = graph.get_operation_by_name(FLAGS.input_layer);

        ground_truth_input = tf.placeholder(
                    tf.float32, [None, dataset.num_classes], name='GroundTruthInput')
        predicts = tf.placeholder(tf.float32, [None, dataset.num_classes], name='predicts')
        accuracy, _ = retrain.add_evaluation_step(predicts, ground_truth_input)

        index = 0

        for i,checkpoint in enumerate(all_checkpoints):
            checkpoint_prefix = checkpoint.replace('.data-00000-of-00001', '')
            saver.restore(sess,checkpoint_prefix)

            predictions = sess.run(output_operation.outputs[0],
                 {input_operation.outputs[0]: ftg})

            #print(predictions.shape)

            feed_dict={predicts: predictions, ground_truth_input: ground_truths}
            ret = accuracy.eval(feed_dict, sess)

            _,fname=os.path.split(checkpoint_prefix)
            prediction_name = fname.replace("model.ckpt-","")

            all_files_df.loc[index,"checkpoint_name"] = fname
            all_files_df.loc[index,"accuracy"] = float(ret)
            index = index + 1

            worksheet.write(i, 0, fname)
            worksheet.write(i, 1, ret)
            print('checkpoint: %s, Accuracy: %g' % (checkpoint,ret))
            sio.savemat(prediction_path+'/prediction_'+prediction_name+'.mat',{"prediction": predictions})

    accuracy_json_name=workspace+"/accuracy/"+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+"accuracy_json.txt"
    saveDataFramToFile(all_files_df, accuracy_json_name)


    stop = time.time()
    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    extract()






