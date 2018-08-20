import os
import shutil

def rename_and_move(old_path, new_path, new_name_prefix):

    file_names = os.listdir(old_path)

    data = []
    for name in file_names:
        if name.find('data') > -1:
            data.append(name)

    prefix = 'model.ckpt-'
    subfix_data = '.data-00000-of-00001'
    subfix_index = '.index'
    new_data = []
    for d in data:
        nd = d
        nd = nd.replace(prefix, '')
        nd = nd.replace(subfix_data, '')
        new_data.append(nd)
    #print(new_data)

    number = []
    for d in new_data:
        new_d = int(d)
        number.append(new_d)
    number.sort()
    print(number)

    for i in range(len(number)):
        new_name_data = prefix + new_name_prefix + str(i) + subfix_data
        #print(new_name_data)
        new_name_index = prefix + new_name_prefix + str(i) + subfix_index

        #os.rename(os.path.join(old_path, prefix+str(number[i])+subfix_data), os.path.join(old_path, new_name_data))
        #os.rename(os.path.join(old_path, prefix+str(number[i])+subfix_index), os.path.join(old_path, new_name_index))
        #shutil.move(os.path.join(old_path, new_name_data), os.path.join(new_path, new_name_data))
        #shutil.move(os.path.join(old_path, new_name_index), os.path.join(new_path, new_name_index))
        print(os.path.join(old_path, prefix+str(number[i])+subfix_data))
        print(os.path.join(new_path, new_name_data))
        shutil.copyfile(os.path.join(old_path, prefix+str(number[i])+subfix_data),os.path.join(new_path, new_name_data))
        shutil.copyfile(os.path.join(old_path, prefix+str(number[i])+subfix_index),os.path.join(new_path, new_name_index))


original_ckpt_dir = './mobilenetv2_on_cifar10_check_point'
new_ckpt_dir = './renamed_check_point'

if os.path.exists(new_ckpt_dir):
    shutil.rmtree(new_ckpt_dir)
os.makedirs(new_ckpt_dir)
parts = os.listdir(original_ckpt_dir)
parts.sort()
print(parts)

for i in range(len(parts)):
    models_dir = os.listdir(os.path.join(original_ckpt_dir, parts[i]))
    models_dir.sort()
    print(models_dir)
    for j in range(len(models_dir)):
        old_path = os.path.join(os.path.join(original_ckpt_dir, parts[i]), models_dir[j])
        rename_and_move(old_path, new_ckpt_dir, str(i)+'-'+str(j)+'-')
