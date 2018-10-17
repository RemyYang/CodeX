# -*- coding:UTF-8 -*-
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
from nets import nets_factory
from datasets import dataset_factory

import pandas as pd
import json
import dataFrame_op

from sklearn.utils import shuffle
import xlsxwriter
import logging
import copy

workspace = "."

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

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './renamed_check_point',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_boolean(
    'base_mode', False,
    'identity if use one base model to ensemble')
FLAGS = tf.app.flags.FLAGS



def logger_setting(logger_path,logger_prefix, logger_level):
  # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("labelimage")

    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    if os.path.exists(logger_path):
        pass
    else:
        os.makedirs(logger_path)

    # 文件日志
    now = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))

    log_file_name = logger_path + "/" + logger_prefix + "_log_" + str(now) + str(".log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter  # 也可以直接给formatter赋值

    # 为logger添加的日志处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if logger_level == "debug":
    # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.DEBUG)
    elif logger_level == "info":
        logger.setLevel(logging.INFO)
    elif logger_level == "warning":
        logger.setLevel(logging.WARNING)
    elif logger_level == "error":
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.CRITICAL)

    logger.debug('This is debug message')
    return logger, file_handler


def loadDataFrameFromJson(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as load_f:
            data_list = json.load(load_f)
            data = [[d["checkpoint_name"],float(d["accuracy"])] for d in data_list["accuracy_list"]]
            all_files_df = pd.DataFrame(data, columns = ["checkpoint_name","accuracy"])
          #logger.debug(all_files_df)
            return all_files_df
    else:
        print("json path is not exist!")
        return 0


#for imagenet
#evaluations1 = {'3-0-0':0.7014200091, '3-1-0':0.7016400099, '3-2-0':0.7019400001, '3-3-0':0.7016199827, '3-4-0':0.7002599835, '3-5-0':0.7012599707, '3-6-0':0.701120019, '3-7-0':0.7016599774, '3-8-0':0.7005400062, '3-9-0':0.7016999722}
def auto_evaluate(bpath, evals, accuracy, subfix='.mat',prefix = "prediction_"):
    size = len(evals)
    #print(len(evals))

    truth = sio.loadmat(truth_mat_name)
    ground_truths = truth['truth']

    pres = []
    for pname in evals:
        pname = pname.replace("model.ckpt-","")
        pname = pname.replace("model.ckpt-","")
        #print(pname)
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


def all_ensemble_strategy(ensemble_df,logger,base_checkpoint_df = None, base_mode = False):

    evals = {}
    checkpoints_accuracy_list = []

    if base_mode == True:
        #print(base_checkpoint_df)
        base_accuracy = base_checkpoint_df.at[0,"accuracy"]
        base_checkpoint_name = base_checkpoint_df.at[0,"checkpoint_name"]
        evals[base_checkpoint_name] = base_accuracy
        checkpoints_accuracy_list.append(base_accuracy)

    for index, row in ensemble_df.iterrows():
        evals[row['checkpoint_name']] = row['accuracy']
        checkpoints_accuracy_list.append(row['accuracy'])

    ensemble_accuracy = auto_evaluate(prediction_path, evals, accuracy = accuracy_eval)

    logger.debug("")

    checkpoints_accuracy_array = np.array(checkpoints_accuracy_list)
    checkpoints_accuracy_mean = np.mean(checkpoints_accuracy_array)
    checkpoints_accuracy_std = np.std(checkpoints_accuracy_array,ddof=1)
    logger.info("checkpoints list accuracy mean is %.5f,  std is %.5f"%(checkpoints_accuracy_mean,checkpoints_accuracy_std))
    logger.info("checkpoints list best accuracy is %.5f"%(max(checkpoints_accuracy_list)))
    logger.info("checkpoints list worst accuracy is %.5f"%(min(checkpoints_accuracy_list)))
    logger.info("checkpoints list all ensemble accuracy is %.5f"%(ensemble_accuracy))

    logger.info("-------------------------------------------------------------")

def greedy_strategy(ensemble_df_origin,logger):
    evals = {}
    pre_accuracy = 0
    ensemble_checkpoints = []
    select_checkpoint_index = 0

    ensemble_df = copy.deepcopy(ensemble_df_origin)

    #logger.debug("ensemble_df before %s"%ensemble_df)


    #count = 0

    while len(ensemble_df) > 0 and select_checkpoint_index!=-1:
        select_checkpoint_index = -1

        #count +=   1
        #logger.debug("count  %s"%count)
        for index, row in ensemble_df.iterrows():
            evals[row['checkpoint_name']] = row['accuracy']
            accuracy = auto_evaluate(prediction_path, evals, accuracy = accuracy_eval)

            #logger.debug("pre_accuracy %s , accuracy  %s"%(pre_accuracy,accuracy))

            if pre_accuracy < accuracy:
                select_checkpoint_index = index
                pre_accuracy = accuracy
            evals.pop(row['checkpoint_name'])

        if select_checkpoint_index != -1:
            #logger.debug("select_checkpoint_index %d"%select_checkpoint_index)
            #print(ensemble_df.loc[[select_checkpoint_index],['checkpoint_name']])
            evals[ensemble_df.loc[select_checkpoint_index,'checkpoint_name']] = ensemble_df.loc[select_checkpoint_index,'accuracy']
            ensemble_checkpoints.append(ensemble_df.loc[select_checkpoint_index,'checkpoint_name'])
            ensemble_df.drop([select_checkpoint_index],inplace=True)

    logger.debug("ensemble_checkpoints nums %d"%len(ensemble_checkpoints))
    logger.debug("ensemble_checkpoints %s"%ensemble_checkpoints)
    logger.debug("ensemble accuracy %s"%pre_accuracy)
    #logger.debug("ensemble_df after %s"%ensemble_df)
    logger.info("-------------------------------------------------------------")



def random_greedy_strategy(ensemble_df,random_times,worksheet,logger, base_checkpoint_df=None,base_mode=False):
    if len(ensemble_df) == 0:
        logger.error("no ensemble checkpoints")
        return

    evals = {}
    pre_accuracy = 0
    ensemble_checkpoints = []

    if base_mode == True:
        #print(base_checkpoint_df)
        base_accuracy = base_checkpoint_df.at[0,"accuracy"]
        base_checkpoint_name = base_checkpoint_df.at[0,"checkpoint_name"]
        logger.debug("base_checkpoint_name is %s, base_accuracy is %.5f"%(base_checkpoint_name,base_accuracy))


        evals[base_checkpoint_name] = base_accuracy
        pre_accuracy = base_accuracy
    ensemble_best_accuracy = 0
    ensemble_worst_accuracy = 1
    ensemble_accuracy_list = []
    ensemble_nums_list = []

    worksheet.write(0, 0, "ensemble accuracy")


    for i in range(random_times):
        evals = {}
        select_ensemble_checkpoints = []
        pre_accuracy = 0
        if base_mode == True:
            evals[base_checkpoint_name] = base_accuracy
            pre_accuracy = base_accuracy
            select_ensemble_checkpoints.append(base_checkpoint_name)

        ensemble_df = shuffle(ensemble_df)

        for index, row in ensemble_df.iterrows():
            evals[row['checkpoint_name']] = row['accuracy']
            accuracy = auto_evaluate(prediction_path, evals, accuracy = accuracy_eval)
            if i == 0:
                ensemble_checkpoints.append(row['checkpoint_name'])
            if accuracy < pre_accuracy:
                evals.pop(row['checkpoint_name'])
                #print('Bad ensemble, removed classifier %s: %g' % (row['checkpoint_name'],row['accuracy']))
            else:
                #print('Classifier %s: %g' % (row['checkpoint_name'], row['accuracy']))
                #print('Ensemble Accuracy: %g' % accuracy)
                pre_accuracy = accuracy
                select_ensemble_checkpoints.append(row['checkpoint_name'])
        ensemble_accuracy_list.append(pre_accuracy)
        ensemble_nums_list.append(len(select_ensemble_checkpoints))
        logger.debug('%d time ensemble,ensemble accuracy is %.5f,  Final ensembled classifiers: %s' % (i+1,pre_accuracy,evals))

        worksheet.write(i+1, 0, pre_accuracy)

        if ensemble_best_accuracy < pre_accuracy:
            ensemble_best_index = i
            ensemble_best_accuracy = pre_accuracy
            best_select_ensemble_checkpoints = list(select_ensemble_checkpoints)
            logger.debug("best_select_ensemble_checkpoints %s"%best_select_ensemble_checkpoints)

        if ensemble_worst_accuracy > pre_accuracy:
            ensemble_worst_index = i
            ensemble_worst_accuracy = pre_accuracy
            worst_select_ensemble_checkpoints = list(select_ensemble_checkpoints)
            logger.debug("best_select_ensemble_checkpoints %s"%worst_select_ensemble_checkpoints)

        logger.debug("")

    if base_mode == True:
        logger.debug("base_checkpoint_name is %s, base_accuracy is %.5f"%(base_checkpoint_name,base_accuracy))
    #logger.debug("ensemble_accuracy_list has %d accuracy"%len(ensemble_accuracy_list))
    logger.info("ensemble best index is %d, ensemble_best_accuracy is %.5f"%(ensemble_best_index+1,ensemble_best_accuracy))
    logger.info("best_select_ensemble_checkpoints %s"%best_select_ensemble_checkpoints)
    logger.info("ensemble worst index is %d, ensemble_worst_accuracy is %.5f"%(ensemble_worst_index+1,ensemble_worst_accuracy))
    logger.info("worst_select_ensemble_checkpoints %s"%worst_select_ensemble_checkpoints)

    ensemble_accuracy_array = np.array(ensemble_accuracy_list)
    ensemble_accuracy_mean = np.mean(ensemble_accuracy_array)
    ensemble_accuracy_std = np.std(ensemble_accuracy_array,ddof=1)
    logger.info("ensemble list accuracy mean is %.5f,  std is %.5f"%(ensemble_accuracy_mean,ensemble_accuracy_std))
    ensemble_nums_array = np.array(ensemble_nums_list)
    ensemble_nums_mean = np.mean(ensemble_nums_array)
    ensemble_nums_std = np.std(ensemble_nums_array,ddof=1)
    logger.info("ensemble list nums mean is %.5f,  std is %.5f"%(ensemble_nums_mean,ensemble_nums_std))

    logger.info("-------------------------------------------------------------")

    if base_mode == True:
        ensemble_checkpoints.append(base_checkpoint_name)

    return ensemble_checkpoints,best_select_ensemble_checkpoints







if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger, file_handler= logger_setting("./log",FLAGS.model_name+'_'+FLAGS.dataset_name+"ensemble","debug")

    accuracy_json_name=workspace+"/"+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+"accuracy_json.txt"
    prediction_path=workspace+'/prediction/'+FLAGS.model_name+'_'+FLAGS.dataset_name
    truth_mat_name=workspace+'/data/'+FLAGS.model_name+'_'+FLAGS.dataset_name+'_'+'truth.mat'


    all_files_df = loadDataFrameFromJson(accuracy_json_name)
    checkpoint_num = len(all_files_df)
    logger.info("checkpoint num is %d"%checkpoint_num)

    if FLAGS.base_mode == True:
        #bese checkpoint is at the last row of the df
        base_checkpoint_df = all_files_df[-1:].reset_index()
        base_accuracy = all_files_df.at[checkpoint_num-1,"accuracy"]
        base_checkpoint_name = all_files_df.at[checkpoint_num-1,"checkpoint_name"]

        all_files_exclude_base_df = all_files_df[0:checkpoint_num-1]
        all_files_exclude_base_df = all_files_exclude_base_df.sort_index(axis=0,ascending=False,by = "accuracy")

        #print(base_accuracy)
        pre_ensemble_df = dataFrame_op.searchDataFromDataFramWithKeyAndBiggerValue(all_files_df[0:checkpoint_num-1],"accuracy",base_accuracy)
        pre_ensemble_num = len(pre_ensemble_df)

        logger.info("base_checkpoint_name is %s, base_accuracy is %.5f"%(base_checkpoint_name,base_accuracy))
        logger.info("above base_accuracy checkpoint num: %d" %(pre_ensemble_num))

    else:
        all_files_df = all_files_df.sort_index(axis=0,ascending=False,by = "accuracy")
        pre_ensemble_num = 50

    #evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
    #random.shuffle(evaluations)

    workbook = xlsxwriter.Workbook("ensemble_accuracy.xlsx")
    worksheet = workbook.add_worksheet()


    with graphs.as_default() as gs:
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ground_truth_input = tf.placeholder(tf.float32, [None, dataset.num_classes], name='GroundTruthInput')
        predictions_average = tf.placeholder(tf.float32, [None, dataset.num_classes], name='predictions_average')

        accuracy_eval, _ = retrain.add_evaluation_step(predictions_average, ground_truth_input)

        logger.info('')

        if pre_ensemble_num < 50:
            pre_ensemble_num =50

        random_times=100
        ensemble_num = int(pre_ensemble_num*0.2)
        logger.info("random ensemble num is : %d , random time is: %d" %(ensemble_num,random_times))
        #sameple will can do the shuffle func

        if FLAGS.base_mode == True:
            pre_ensemble_df = all_files_exclude_base_df[0:ensemble_num]
            ensemble_df = pre_ensemble_df[:].sample(n = ensemble_num)
            logger.info('')
            logger.info('all checkpoints ensemble.......')
            all_ensemble_strategy(ensemble_df, logger,base_checkpoint_df = base_checkpoint_df,base_mode = FLAGS.base_mode)

            logger.info('Random greedy.......')
            ensemble_checkpoints,select_ensemble_checkpoints=random_greedy_strategy(ensemble_df,random_times,worksheet,logger,base_checkpoint_df = base_checkpoint_df,base_mode = FLAGS.base_mode)

        else:
            pre_ensemble_df = all_files_df[0:ensemble_num]
            ensemble_df = pre_ensemble_df[:].sample(n = ensemble_num)
            logger.info('')
            ####################all_ensemble_strategy
            logger.info('all checkpoints ensemble.......')
            all_ensemble_strategy(ensemble_df, logger)

            ####################greedy strategy
            logger.info('greedy.......')

            greedy_strategy(ensemble_df,logger)
            print(ensemble_df)


            ####################random greedy
            logger.info('Random greedy.......')

            random_times_list = [20,40,60,80,100]
            for times in random_times_list:
                logger.info('#################Random times %d#################'%times)
                ensemble_checkpoints,select_ensemble_checkpoints=random_greedy_strategy(ensemble_df,times,worksheet,logger)


#        weights,biases=pca.average(ensemble_checkpoints,select_ensemble_checkpoints, dataset,FLAGS)
#        assert len(weights) == len(biases)
#        parameters = []
#        for i in range(len(biases)):
#            param = pca.get_parameters_w(weights[i][0][0])
#            param += biases[i].tolist()
#            parameters.append(param)
#        X = np.array(parameters)
#        estimator = PCA(n_components=2)
#        new_params = estimator.fit_transform(X)
#        pca.write_excel(new_params)



#        print('')
#        print('Pattern MAX........')
#        evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = True)
#        evals = {}
#        pre_accuracy = 0
#        for item in evaluations:
#            evals[item[0]] = item[1]
#            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)#

#            if accuracy < pre_accuracy:
#                evals.pop(item[0])
#                print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
#            else:
#                print('Classifier %s: %g' % (item[0], item[1]))
#                print('Ensemble Accuracy: %g' % accuracy)
#                pre_accuracy = accuracy
#        print('Final ensembled classifiers: %s' % evals)#

#        print('')
#        print('Pattern MIN........')
#        evaluations = sorted(evaluations1.items(),key = lambda x:x[1],reverse = False)
#        evals = {}
#        pre_accuracy = 0
#        for item in evaluations:
#            evals[item[0]] = item[1]
#            accuracy = auto_evaluate('./prediction', evals, accuracy = accuracy_eval)#

#            if accuracy < pre_accuracy:
#                evals.pop(item[0])
#                print('Bad ensemble, removed classifier %s: %g' % (item[0], item[1]))
#            else:
#                print('Classifier %s: %g' % (item[0], item[1]))
#                print('Ensemble Accuracy: %g' % accuracy)
#                pre_accuracy = accuracy
#        print('Final ensembled classifiers: %s' % evals)


