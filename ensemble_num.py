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

import random

graphs = tf.Graph()

'''
evaluations2 = {'prediction_304':0.7015600204, 'prediction_314':0.7018200159, 'prediction_324':0.7018200159, 'prediction_334':0.7017199993, 'prediction_344':0.7002800107, 'prediction_354':0.7010400295, 'prediction_364':0.7011600137, 'prediction_374':0.7015600204, 'prediction_384':0.7006199956, 'prediction_394':0.7016800046}

evaluations10 = {'prediction_303':0.7014799714, 'prediction_313':0.7015200257, 'prediction_323':0.7019199729, 'prediction_333':0.7018600106, 'prediction_343':0.7002000213, 'prediction_353':0.7010800242, 'prediction_363':0.7012000084, 'prediction_373':0.7015399933, 'prediction_383':0.7005400062, 'prediction_393':0.7016199827}

evaluations3 = {'prediction_302':0.701399982, 'prediction_312':0.7016199827, 'prediction_322':0.7020199895, 'prediction_332':0.7016199827, 'prediction_342':0.7003600001, 'prediction_352':0.7011799812, 'prediction_362':0.701300025, 'prediction_372':0.701579988, 'prediction_382':0.7005400062, 'prediction_392':0.7017999887}

evaluations1 = {'prediction_301':0.7011600137, 'prediction_311':0.701839, 'prediction_321':0.7020999789, 'prediction_331':0.7017400265, 'prediction_341':0.7002000213, 'prediction_351':0.701399982, 'prediction_361':0.7013599873, 'prediction_371':0.7017999887, 'prediction_381':0.7005000114, 'prediction_391':0.7020199895}
'''
evaluations1 = {'prediction_300':0.7014200091, 'prediction_310':0.7016400099, 'prediction_320':0.7019400001, 'prediction_330':0.7016199827, 'prediction_340':0.7002599835, 'prediction_350':0.7012599707, 'prediction_360':0.701120019, 'prediction_370':0.7016599774, 'prediction_380':0.7005400062, 'prediction_390':0.7016999722}


def auto_evaluate(bpath, evals, subfix='.mat', cls=1001):
    size = len(evals)

    with graphs.as_default() as gs:
        ground_truth_input = tf.placeholder(tf.float32, [None, cls], name='GroundTruthInput')
        predictions = []
        for i in range(size):
            temp = tf.placeholder(tf.float32, [None, cls], name='pre'+str(i))
            predictions.append(temp)
        accuracy, _ = retrain.add_evaluation_step(sum(predictions), ground_truth_input)

    truth_path = os.path.join('./data', 'truth'+subfix)
    truth = sio.loadmat(truth_path)
    ground_truths = truth['truth']

    pres = []
    for pname in evals:
        pre = sio.loadmat(os.path.join(bpath, pname+subfix))
        pres.append(pre)
    pres_value = []
    for pre in pres:
        pres_value.append(pre['prediction'])

    accuracies = []
    i = 0
    with tf.Session(graph=gs) as sess:
        for i in range(len(ground_truths)):
            feed_dict = {}
            for j in range(size):
                feed_dict[predictions[j]] = pres_value[j][i]
            feed_dict[ground_truth_input] = ground_truths[i]

            accuracies.append(accuracy.eval(feed_dict, sess))

    return np.mean(accuracies)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
    random.shuffle(evaluations)

    print('Random.....')
    print(evaluations)
    evals = {}
    pre_accuracy = 0
    for item in evaluations:
        evals[item[0]] = item[1]
        accuracy = auto_evaluate('./prediction', evals)
        print('Classifier %s: %g' % (item[0], item[1]))
        print('Ensemble Accuracy: %g' % accuracy)
    print('Final ensembled classifiers: %s' % evals)

    print('')
    print('Random greedy.......')
    evals = {}
    pre_accuracy = 0
    for item in evaluations:
        evals[item[0]] = item[1]
        accuracy = auto_evaluate('./prediction', evals)
        if accuracy < pre_accuracy:
            evals.pop(item[0])
            print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
        else:
            print('Classifier %s: %g' % (item[0], item[1]))
            print('Ensemble Accuracy: %g' % accuracy)
            pre_accuracy = accuracy
    print('Final ensembled classifiers: %s' % evals)

    print('')
    print('Pattern MAX........')
    evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
    evals = {}
    pre_accuracy = 0
    for item in evaluations:
        evals[item[0]] = item[1]
        accuracy = auto_evaluate('./prediction', evals)

        if accuracy < pre_accuracy:
            evals.pop(item[0])
            print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
        else:
            print('Classifier %s: %g' % (item[0], item[1]))
            print('Ensemble Accuracy: %g' % accuracy)
            pre_accuracy = accuracy
    print('Final ensembled classifiers: %s' % evals)

    print('')
    print('Pattern MIN........')
    evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = False)
    evals = {}
    pre_accuracy = 0
    for item in evaluations:
        evals[item[0]] = item[1]
        accuracy = auto_evaluate('./prediction', evals)

        if accuracy < pre_accuracy:
            evals.pop(item[0])
            print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
        else:
            print('Classifier %s: %g' % (item[0], item[1]))
            print('Ensemble Accuracy: %g' % accuracy)
            pre_accuracy = accuracy
    print('Final ensembled classifiers: %s' % evals)


