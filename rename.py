import os
import shutil
import tensorflow as tf
from glob import glob
tf.app.flags.DEFINE_string(
    'original_ckpt_dir', './mobilenetv2_on_cifar10_check_point', 'original_ckpt_dir')
tf.app.flags.DEFINE_string(
    'new_ckpt_dir', './renamed_check_point', 'new_ckpt_dir')
tf.app.flags.DEFINE_integer(
    'select_model_num', 10, 'ever model dir selected num')
tf.app.flags.DEFINE_boolean(
    'base_mode', False,
    'identity if use one base model to ensemble')

FLAGS = tf.app.flags.FLAGS

def rename_and_copy(old_path, new_path, new_name_prefix):

    file_names = os.listdir(old_path)

    data_list = glob(os.path.join(old_path, "*.data*"))
    if data_list != []:
        data_list.sort(reverse=True)

        prefix = 'model.ckpt-'
        subfix_data = '.data-00000-of-00001'
        subfix_index = '.index'

        for i, file_path in enumerate(data_list):
            if i == FLAGS.select_model_num:
                break
            index_path = file_path.replace(subfix_data,subfix_index)
            new_name_data = prefix + new_name_prefix + str(i) + subfix_data
            new_name_index = prefix + new_name_prefix + str(i) + subfix_index

            shutil.copyfile(file_path,os.path.join(new_path, new_name_data))
            shutil.copyfile(index_path,os.path.join(new_path, new_name_index))

    data_list = glob(os.path.join(old_path, "*.ckpt"))
    if data_list != []:
        data_list.sort(reverse=True)
        prefix = 'model'
        old_subfix_ckpt = '.ckpt'
        for i, file_path in enumerate(data_list):
            new_name_ckpt = prefix + new_name_prefix + str(i) + old_subfix_ckpt
            shutil.copyfile(file_path,os.path.join(new_path, new_name_ckpt))

""


#    data = []
#    for name in file_names:
#        if name.find('data') > -1:
#            data.append(name)

#    prefix = 'model.ckpt-'
#    subfix_data = '.data-00000-of-00001'
#    subfix_index = '.index'
#    new_data = []
#    for d in data:
#        nd = d
#        nd = nd.replace(prefix, '')
#        nd = nd.replace(subfix_data, '')
#        new_data.append(nd)
#    #print(new_data)#

#    number = []
#    for d in new_data:
#        new_d = int(d)
#        number.append(new_d)
#    number.sort()
#    #print(number)#

#    for i in range(len(number)):
#        new_name_data = prefix + new_name_prefix + str(i) + subfix_data
#        #print(new_name_data)
#        new_name_index = prefix + new_name_prefix + str(i) + subfix_index#

#        #os.rename(os.path.join(old_path, prefix+str(number[i])+subfix_data), os.path.join(old_path, new_name_data))
#        #os.rename(os.path.join(old_path, prefix+str(number[i])+subfix_index), os.path.join(old_path, new_name_index))
#        #shutil.move(os.path.join(old_path, new_name_data), os.path.join(new_path, new_name_data))
#        #shutil.move(os.path.join(old_path, new_name_index), os.path.join(new_path, new_name_index))
#        #print(os.path.join(old_path, prefix+str(number[i])+subfix_data))
#        #print(os.path.join(new_path, new_name_data))
#        shutil.copyfile(os.path.join(old_path, prefix+str(number[i])+subfix_data),os.path.join(new_path, new_name_data))
#        shutil.copyfile(os.path.join(old_path, prefix+str(number[i])+subfix_index),os.path.join(new_path, new_name_index))

if __name__ == "__main__":
    if os.path.exists(FLAGS.new_ckpt_dir):
        shutil.rmtree(FLAGS.new_ckpt_dir)
    os.makedirs(FLAGS.new_ckpt_dir)
    parts = os.listdir(FLAGS.original_ckpt_dir)
    parts.sort()
    print(parts)

    for i in range(len(parts)):
        if parts[i] != "base":
            models_dir = os.listdir(os.path.join(FLAGS.original_ckpt_dir, parts[i]))
            models_dir.sort()
            print(models_dir)
            for j in range(len(models_dir)):
                old_path = os.path.join(os.path.join(FLAGS.original_ckpt_dir, parts[i]), models_dir[j])
                rename_and_copy(old_path, FLAGS.new_ckpt_dir, str(i)+'-'+str(j)+'-')
        else:
            if FLAGS.base_mode == True:
                old_path = os.path.join(FLAGS.original_ckpt_dir, parts[i])
                rename_and_copy(old_path, FLAGS.new_ckpt_dir, "best-")
