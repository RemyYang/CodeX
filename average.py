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

import time

from tensorflow.python.framework.graph_util import convert_variables_to_constants

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





