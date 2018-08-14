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

graphs = tf.Graph()

truth = sio.loadmat('/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/data/truth.mat')
ground_truths = truth['truth']

def evaluate(pre_mat_path):

    with graphs.as_default() as gs:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, 1001], name='GroundTruthInput')
        fts = tf.placeholder(tf.float32, [None, 1001], name='fts')
        accuracy, _ = retrain.add_evaluation_step(fts, ground_truth_input)

    pre = sio.loadmat(pre_mat_path)
    pre = pre['prediction']

    accuracies = []
    i = 0
    with tf.Session(graph=gs) as sess:
        for ft, ground_truth in zip(pre, ground_truths):
            feed_dict={fts: ft, ground_truth_input: ground_truth}
            accuracies.append(accuracy.eval(feed_dict, sess))

    return np.mean(accuracies)


import xlsxwriter
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    pred_path = '/home/xxxx/PHICOMM/RemyWorkSpace/rename_file/prediction'
    preds = os.listdir(pred_path)
    predictions = []
    for pr in preds:
        if pr.find('prediction') > -1:
            predictions.append(pr)
    print(predictions)
    print(len(predictions))

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


