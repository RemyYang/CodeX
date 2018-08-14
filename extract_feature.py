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

import scipy.io as sio

graph = load_graph('./base/base.pb')

def extract():

    with graph.as_default() as g:
        image_buffer_input = g.get_tensor_by_name('input:0')
        final_tensor = g.get_tensor_by_name('MobilenetV2/Logits/AvgPool:0')

    image_dir = '/home/xxxx/PHICOMM/ai-share/dataset/imagenet/raw-data/imagenet-data/validation'
    testing_percentage = 100
    validation_percentage = 0
    category='testing'

    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage,
        validation_percentage)
    class_count = len(image_lists.keys())
    
    ground_truths = []
    filenames = []
        
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(image_lists[label_name][category]):
        image_name = retrain.get_image_path(
            image_lists, label_name, image_index, image_dir, category)
        ground_truth = np.zeros([1, class_count+1], dtype=np.float32)
        ground_truth[0, label_index+1] = 1.0
        ground_truths.append(ground_truth)
        filenames.append(image_name)

    sio.savemat('./data/truth.mat',{"truth": ground_truths})

    cf = 0.875
    
    ftg = []
    i = 0
    start = time.time()
    with tf.Session(graph=g) as sess:
        for filename in filenames:    
            image = Image.open(filename).convert("RGB")
            #print("filename is %s"%filename)
            #print("This image has %d channels."%image.layers)
            x0 = int(image.size[0]*(1-cf)/2)
            y0 = int(image.size[1]*(1-cf)/2)
            x1 = x0 + int(image.size[0]*cf)
            y1 = y0 + int(image.size[1]*cf)
            image = image.crop([x0,y0,x1,y1])
            image = image.resize((224,224),Image.ANTIALIAS)
            image = np.array(image, dtype=np.float32)[None,...]
            image = (image-128)/128.0   
            feed_dict={image_buffer_input: image}
            ft = final_tensor.eval(feed_dict, sess)
            ftg.append(ft)
            i = i + 1
            print(i)
    stop = time.time()
    print(str((stop-start)/len(ftg))+' seconds.')

    sio.savemat('./data/feature.mat',{"feature": ftg})


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    extract()




  

