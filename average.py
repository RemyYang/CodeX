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

from sklearn.decomposition import PCA
from glob import glob

import time

from tensorflow.python.framework.graph_util import convert_variables_to_constants
sys.path.append("/home/deepl/PHICOMM/FoodAI/FoodAi/tensorflow/tensorflow_models/models/research/PHICOMM/slim")
from nets import nets_factory




def get_graph(path):

    clss = os.listdir(path)
    clss.sort()
    print(clss)
    graphs = []
    for cls in clss:
        graph_classifier = load_graph(os.path.join(path, cls))
        graphs.append(graph_classifier)

    return graphs

def get_weights_biases(graphs, allw, selective):

    weights = []
    biases = []
    for graph in graphs:
        with graph.as_default() as g:
            weight = graph.get_tensor_by_name('MobilenetV2/Logits/Conv2d_1c_1x1/weights:0')
            biase = graph.get_tensor_by_name('MobilenetV2/Logits/Conv2d_1c_1x1/biases:0')

        with tf.Session(graph=g) as sess:
            ws = weight.eval({}, sess)
            bs = biase.eval({}, sess)

        weights.append(ws)
        biases.append(bs)

    mean_w = weights[allw[0]].copy()
    mean_b = biases[allw[0]].copy()
    for i in range(len(allw)):
        mean_w += weights[allw[i]]
        mean_b += biases[allw[i]]
    mean_w -= weights[allw[0]]
    mean_b -= biases[allw[0]]
    mean_w /= (len(allw)*1.0)
    mean_b /= (len(allw)*1.0)
    weights.append(mean_w)
    biases.append(mean_b)

    mean_w = weights[selective[0]].copy()
    mean_b = biases[selective[0]].copy()
    for i in range(len(selective)):
        mean_w += weights[selective[i]]
        mean_b += biases[selective[i]]
    mean_w -= weights[selective[0]]
    mean_b -= biases[selective[0]]
    mean_w /= (len(selective)*1.0)
    mean_b /= (len(selective)*1.0)
    weights.append(mean_w)
    biases.append(mean_b)

    return weights, biases


def get_parameters_w(weights):

    w = []
    weights = weights.transpose()
    for weight in weights:
        w.append(np.mean(weight))

    return w

import xlsxwriter
def write_excel(new_params):

    workbook = xlsxwriter.Workbook("parameters.xlsx")
    worksheet = workbook.add_worksheet()

    for i in range(len(new_params)):
        for j in range(len(new_params[0])):
            worksheet.write(i, j, new_params[i][j])

    workbook.close()

def get_weight_and_bias(checkpoint_prefix,sess,graph,saver):
    saver.restore(sess,checkpoint_prefix)
    weight = graph.get_tensor_by_name('MobilenetV2/Logits/Conv2d_1c_1x1/weights:0')
    biase = graph.get_tensor_by_name('MobilenetV2/Logits/Conv2d_1c_1x1/biases:0')
    ws = weight.eval({}, sess)
    bs = biase.eval({}, sess)
    return ws,bs


def average(ensemble_checkpoints,select_ensemble_checkpoints,dataset,FLAGS):
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
    sess = tf.Session()
    weights = []
    biases = []
    weights_tmp = []
    biases_tmp = []

    all_checkpoints = glob(os.path.join(FLAGS.checkpoint_path, "*.data*"))
    all_checkpoints.sort()
    #print(all_checkpoints)

    #total checkpoints
    for i,checkpoint in enumerate(all_checkpoints):
        #print(checkpoint)
        checkpoint_prefix = checkpoint.replace('.data-00000-of-00001', '')
        #print(checkpoint_prefix)
        ws,bs = get_weight_and_bias(checkpoint_prefix, sess, graph, saver)
        weights.append(np.array(ws))
        biases.append(np.array(bs))

    #print(ensemble_checkpoints)

    for i,checkpoint in enumerate(ensemble_checkpoints):
        checkpoint = checkpoint.replace("model.ckpt-","")
        checkpoint_prefix = FLAGS.checkpoint_path + "/" + "model.ckpt-" +checkpoint
        #print(checkpoint_prefix)
        ws,bs = get_weight_and_bias(checkpoint_prefix, sess, graph, saver)
        weights_tmp.append(np.array(ws))
        biases_tmp.append(np.array(bs))

    weights_tmp_stack = np.stack(weights_tmp,0)
    biases_tmp_stack = np.stack(biases_tmp,0)
    weights.append(np.array(np.mean(weights_tmp_stack,axis = 0)))
    biases.append(np.array(np.mean(biases_tmp_stack,axis = 0)))

    weights_tmp = []
    biases_tmp = []

    for i,checkpoint in enumerate(select_ensemble_checkpoints):
        checkpoint = checkpoint.replace("model.ckpt-","")
        checkpoint_prefix = FLAGS.checkpoint_path + "/" + "model.ckpt-" +checkpoint

        #print(checkpoint_prefix)
        ws,bs = get_weight_and_bias(checkpoint_prefix, sess, graph, saver)
        weights_tmp.append(np.array(ws))
        biases_tmp.append(np.array(bs))

    weights_tmp_stack = np.stack(weights_tmp,0)
    biases_tmp_stack = np.stack(biases_tmp,0)
    weights.append(np.array(np.mean(weights_tmp_stack,axis = 0)))
    biases.append(np.array(np.mean(biases_tmp_stack,axis = 0)))
    return weights,biases






if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    graphs = get_graph('./classifier')
    allw_idx = [3*50+4*5+4,3*50+6*5+4,3*50+7*5+4,3*50+8*5+4,3*50+3*5+4,3*50+0*5+4,3*50+1*5+4,3*50+9*5+4,3*50+5*5+4,3*50+2*5+4]
    selective_idx = [3*50+4*5+4,3*50+6*5+4,3*50+3*5+4,3*50+0*5+4,3*50+1*5+4,3*50+9*5+4,3*50+5*5+4,3*50+2*5+4]
    weights, get_weights_biases = get_weights_biases(graphs, allw_idx, selective_idx)
    assert len(weights) == len(biases)

    parameters = []
    for i in range(len(biases)):
        param = get_parameters_w(weights[i][0][0])
        param += biases[i].tolist()
        parameters.append(param)

    X = np.array(parameters)
    estimator = PCA(n_components=2)
    new_params = estimator.fit_transform(X)
    write_excel(new_params)





