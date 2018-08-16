#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import argparse

import numpy as np
import PIL.Image as Image
import tensorflow as tf

import retrain as retrain
from count_ops import load_graph

import time
import shutil

import scipy.io as sio

workspace="/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX"

def generate_prediction(graph_classifier, name):

    with graph_classifier.as_default() as gc:
        feature_input = graph_classifier.get_tensor_by_name('MobilenetV2/Logits/AvgPool:0')
        predict = graph_classifier.get_tensor_by_name('MobilenetV2/Predictions/Reshape_1:0')

    feature = sio.loadmat(workspace+'/data/feature.mat')
    truth = sio.loadmat(workspace+'/data/truth.mat')
    ftg = feature['feature']
    ground_truths = truth['truth']

    predictions = []
    i = 0
    start = time.time()
    with tf.Session(graph=gc) as sess:
        for f in ftg:
            #print(i)
            i = i + 1
            feed_dict={feature_input: f}
            predictions.append(predict.eval(feed_dict, sess))
    stop = time.time()

    print(str((stop-start)/len(predictions))+' seconds.')

    sio.savemat(workspace+'/prediction/prediction_'+name+'.mat',{"prediction": predictions})


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    pb_path =workspace+'/classifier'
    prediction_path=workspace+'/prediction'

    if os.path.exists(prediction_path):
        print("%s is exist, please delete it!"%prediction_path)
        exit()
        #shutil.rmtree(prediction_path)
    os.makedirs(prediction_path)

    total_start = time.time()

    pbs = os.listdir(pb_path)
    classifiers = []
    for pb in pbs:
        if pb.find('classifier') > -1:
            classifiers.append(pb)
    print(len(classifiers))

    cls_nums = []
    for cls in classifiers:
        ncls = cls
        ncls = ncls.replace('classifier_', '')
        ncls = ncls.replace('.pb', '')
#        if ncls[0]=='3':
        cls_nums.append(ncls)
    print(cls_nums)

    i = 0
    for cn in cls_nums:
        print('Predicting.....num=%d'%i)
        i += 1
        print(os.path.join(pb_path, 'classifier_'+cn+'.pb'))
        graph_classifier = load_graph(os.path.join(pb_path, 'classifier_'+cn+'.pb'))
        generate_prediction(graph_classifier, cn)

    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')
    print('Generating predictions finished.')






