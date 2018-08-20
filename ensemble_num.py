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
import average as pca

import time
from glob import glob

import scipy.io as sio

import random
from sklearn.decomposition import PCA
graphs = tf.Graph()
sys.path.append("/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim")
from nets import nets_factory
from datasets import dataset_factory

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', "/home/deepl/PHICOMM/dataset/cifar10_tf/cifar10_test.tfrecord", 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './renamed_check_point',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')


FLAGS = tf.app.flags.FLAGS

'''
evaluations2 = {'prediction_304':0.7015600204, 'prediction_314':0.7018200159, 'prediction_324':0.7018200159, 'prediction_334':0.7017199993, 'prediction_344':0.7002800107, 'prediction_354':0.7010400295, 'prediction_364':0.7011600137, 'prediction_374':0.7015600204, 'prediction_384':0.7006199956, 'prediction_394':0.7016800046}

evaluations10 = {'prediction_303':0.7014799714, 'prediction_313':0.7015200257, 'prediction_323':0.7019199729, 'prediction_333':0.7018600106, 'prediction_343':0.7002000213, 'prediction_353':0.7010800242, 'prediction_363':0.7012000084, 'prediction_373':0.7015399933, 'prediction_383':0.7005400062, 'prediction_393':0.7016199827}

evaluations3 = {'prediction_302':0.701399982, 'prediction_312':0.7016199827, 'prediction_322':0.7020199895, 'prediction_332':0.7016199827, 'prediction_342':0.7003600001, 'prediction_352':0.7011799812, 'prediction_362':0.701300025, 'prediction_372':0.701579988, 'prediction_382':0.7005400062, 'prediction_392':0.7017999887}

evaluations1 = {'prediction_301':0.7011600137, 'prediction_311':0.701839, 'prediction_321':0.7020999789, 'prediction_331':0.7017400265, 'prediction_341':0.7002000213, 'prediction_351':0.701399982, 'prediction_361':0.7013599873, 'prediction_371':0.7017999887, 'prediction_381':0.7005000114, 'prediction_391':0.7020199895}
'''
evaluations1 = {'0-0-34':0.918200016,'0-1-34':0.9179999828, '0-2-34':0.918500006198883, '0-3-34':0.917599976062775, '0-4-35':0.918900012969971, '0-5-34':0.919200003147125, '0-6-34':0.917999982833862, '0-7-34':0.918699979782104, '0-8-35':0.917999982833862, '0-9-35':0.918600022792816}


def auto_evaluate(bpath, evals, accuracy, subfix='.mat',prefix = "prediction_"):
    size = len(evals)
    #print(len(evals))


    truth_path = os.path.join('./data', 'truth'+subfix)
    truth = sio.loadmat(truth_path)
    ground_truths = truth['truth']

    pres = []
    for pname in evals:
        pre = sio.loadmat(os.path.join(bpath, prefix+pname+subfix))
        pres.append(pre)
    pres_value = []
    for pre in pres:
        pres_value.append(np.array(pre['prediction']))

    pres_value_stack = np.stack(pres_value,0)

    #print(pres_value_stack.shape)
    #print(len(ground_truths))

    feed_dict = {}
    feed_dict[predictions_average] = np.mean(pres_value_stack,axis = 0)
    feed_dict[ground_truth_input] = ground_truths

    sess = tf.Session()
    accuracies = accuracy.eval(feed_dict, sess)
    #print(accuracies)

    return accuracies


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
    random.shuffle(evaluations)

    with graphs.as_default() as gs:
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ground_truth_input = tf.placeholder(tf.float32, [None, dataset.num_classes], name='GroundTruthInput')
        predictions_average = tf.placeholder(tf.float32, [None, dataset.num_classes], name='predictions_average')

        accuracy_eval, _ = retrain.add_evaluation_step(predictions_average, ground_truth_input)

        print(evaluations)
        print('Random.....')
        evals = {}
        pre_accuracy = 0
        for item in evaluations:
            evals[item[0]] = item[1]
            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)
            print('Classifier %s: %g' % (item[0], item[1]))
            print('Ensemble Accuracy: %g' % accuracy)
        print('Final ensembled classifiers: %s' % evals)

        print('')
        print('Random greedy.......')
        evals = {}
        pre_accuracy = 0
        ensemble_checkpoints = []
        select_ensemble_checkpoints = []
        for item in evaluations:
            print(item[0],item[1])
            evals[item[0]] = item[1]
            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)

            ensemble_checkpoints.append(item[0])
            if accuracy < pre_accuracy:
                evals.pop(item[0])
                print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
            else:
                print('Classifier %s: %g' % (item[0], item[1]))
                print('Ensemble Accuracy: %g' % accuracy)
                pre_accuracy = accuracy
                select_ensemble_checkpoints.append(item[0])

        print('Final ensembled classifiers: %s' % evals)

        weights,biases=pca.average(ensemble_checkpoints,select_ensemble_checkpoints, dataset,FLAGS)
        assert len(weights) == len(biases)

        parameters = []
        for i in range(len(biases)):
            param = pca.get_parameters_w(weights[i][0][0])
            param += biases[i].tolist()
            parameters.append(param)

        X = np.array(parameters)
        estimator = PCA(n_components=2)
        new_params = estimator.fit_transform(X)
        pca.write_excel(new_params)



        print('')
        print('Pattern MAX........')
        evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
        evals = {}
        pre_accuracy = 0
        for item in evaluations:
            evals[item[0]] = item[1]
            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)

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
            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)

            if accuracy < pre_accuracy:
                evals.pop(item[0])
                print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
            else:
                print('Classifier %s: %g' % (item[0], item[1]))
                print('Ensemble Accuracy: %g' % accuracy)
                pre_accuracy = accuracy
        print('Final ensembled classifiers: %s' % evals)


