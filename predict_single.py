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

import scipy.io as sio

slim = tf.contrib.slim
graphs = tf.Graph()

workspace="/home/deepl/PHICOMM/RemyWorkSpace/ensemble/CodeX"

truth = sio.loadmat(workspace+'/data/truth.mat')
ground_truths = truth['truth']
#print(ground_truths.shape)
ground_truths = np.squeeze(ground_truths)
#print(ground_truths.shape)




def evaluate(pre_mat_path):

    with graphs.as_default() as gs:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, 10], name='GroundTruthInput')
        fts = tf.placeholder(tf.float32, [None, 10], name='fts')
        accuracy, _ = retrain.add_evaluation_step(fts, ground_truth_input)

    pre = sio.loadmat(pre_mat_path)
    pre = pre['prediction']
    pre = np.squeeze(pre)
    #print(pre.shape)
    accuracies = []
    i = 0
 #   with tf.Session(graph=gs) as sess:
 #       for ft, ground_truth in zip(pre, ground_truths):
 #           #print(ft)
 #           #print(ground_truth)
 #           feed_dict={fts: ft, ground_truth_input: ground_truth}
 #           accuracies.append(accuracy.eval(feed_dict, sess))
     #return np.mean(accuracies)
    with tf.Session(graph=gs) as sess:
            #print(ft)
            #print(ground_truth)
        feed_dict={fts: pre, ground_truth_input: ground_truths}
        #accuracies.append(accuracy.eval(feed_dict, sess))
        ret = accuracy.eval(feed_dict, sess)

    return ret




import xlsxwriter
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    pred_path = workspace+'/prediction'
    preds = os.listdir(pred_path)
    predictions = []
    for pr in preds:
        if pr.find('prediction') > -1:
            predictions.append(pr)
    print(predictions)
    print(len(predictions))

    total_start = time.time()


    pre_nums = []
    for pre in predictions:
        npre = pre
        npre = pre.replace('prediction_', '')
        npre = npre.replace('.mat', '')
#        if npre[0]=='3':
#            pre_nums.append(npre)
        pre_nums.append(npre)
    pre_nums.sort()
    print(pre_nums)
    print(len(pre_nums))

    workbook = xlsxwriter.Workbook("accuracies3.xlsx")
    worksheet = workbook.add_worksheet()
    for i in range(len(pre_nums)):
        print('Evaluating.....')
        pre_mat_path = os.path.join(pred_path, 'prediction_'+pre_nums[i]+'.mat')
        accuracy = evaluate(pre_mat_path)
        worksheet.write(i, 0, pre_nums[i])
        worksheet.write(i, 1, accuracy)
        print('Ensemble Accuracy: %g' % accuracy)
    workbook.close()

    total_stop = time.time()
    print("total time is "+str((total_stop-total_start))+' seconds.')


