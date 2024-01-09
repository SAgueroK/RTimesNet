import os
import shutil
import time

from sklearn.model_selection import KFold

import config
import torch
import numpy as np
from ptflops import get_model_complexity_info

HP = config.HP
factor = 0.2


def sample(data):
    data = torch.tensor(data[::HP.sample_step, :])
    return data


def save_checkpoint(model_, optimizer, checkpoint_path):
    save_dict = {
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(save_dict, checkpoint_path)


def MAE(pred, true):
    pred = pred.to('cpu')
    true = true.to('cpu')
    return np.mean((np.abs(pred - true)).numpy())


def MSE(pred, true):
    pred = pred.to('cpu')
    true = true.to('cpu')
    return np.mean(((pred - true) ** 2).numpy())


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print(f'参数大小为：{param_size / 1024 / 1024}MB')
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def del_dir(path_file):
    os.remove(path_file)


def copyfile(origin_file, destination_path):  # 复制函数:将origin_file文件复制到destination_path中
    destination_path = destination_path + '/'  # destination_path要加/
    if not os.path.isfile(origin_file):
        print("%s not exist!" % origin_file)
    else:
        f_path, f_name = os.path.split(origin_file)  # 分离文件名和路径
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)  # 创建路径
        shutil.copy(origin_file, destination_path + f_name)  # 复制文件
        print("copy %s -> %s" % (origin_file, destination_path + f_name))


def get_all_file_path(path):  # 获取path下的所有文件，并且返回文件项目相对路径,例：./datasets/origin/false/2023-05-0812.12.47
    if os.path.exists(path):
        files = os.listdir(path)
        for i, file in enumerate(files):
            files[i] = path + '/' + file
        return files
    else:
        return []


# 数据集初始化，将所有初试文件打乱后按6:2:2分配给训练，验证和测试集
def data_init():
    # 获取训练 测试 验证 文件路径
    HP = config.HP
    filePath_origin_true = HP.data_dir_origin_true
    filePath_origin_false = HP.data_dir_origin_false
    filePath_train_true = HP.data_dir_train_true
    filePath_train_false = HP.data_dir_train_false
    filePath_test_true = HP.data_dir_test_true
    filePath_test_false = HP.data_dir_test_false
    acc_path = HP.data_dir_acc
    ori_path = HP.data_dir_ori
    gyro_path = HP.data_dir_gyro

    all_user = get_all_file_path(HP.data_dir)
    for user in all_user:
        delete_datasets(user)
        # 获取所有文件
        allocation_datasets(user, user + '/' + filePath_origin_true, user + '/' + filePath_origin_false,
                            user + '/' + filePath_train_true, user + '/' + filePath_train_false,
                            user + '/' + filePath_test_true, user + '/' + filePath_test_false)


# 删除某个文件夹下的所有数据集
def delete_datasets(path):
    if os.path.exists(path + '/test'):
        del_files(path + '/test')
    if os.path.exists(path + '/train'):
        del_files(path + '/train')

# 分配数据
def allocation_datasets(user, filePath_origin_true, filePath_origin_false, filePath_train_true,
                        filePath_train_false, filePath_test_true, filePath_test_false):
    if os.path.exists(filePath_train_true):
        shutil.rmtree(filePath_train_true)
    if os.path.exists(filePath_train_false):
        shutil.rmtree(filePath_train_false)
    if os.path.exists(filePath_test_true):
        shutil.rmtree(filePath_test_true)
    if os.path.exists(filePath_test_false):
        shutil.rmtree(filePath_test_false)
    # 获取所有文件
    fileList_x_true = get_all_file_path(filePath_origin_true)
    fileList_x_false_origin = get_all_file_path(filePath_origin_false)
    kf = KFold(n_splits=HP.K_NUM)
    k_num = 1
    print(filePath_origin_true)
    for X_train, X_test in kf.split(fileList_x_true):
        for index in X_train:
            copyfile(fileList_x_true[index], filePath_train_true + '/' + str(k_num))
        for index in X_test:
            copyfile(fileList_x_true[index], filePath_test_true + '/' + str(k_num))
        k_num += 1
    k_num = 1
    for X_train, X_test in kf.split(fileList_x_false_origin):
        for index in X_train:
            copyfile(fileList_x_false_origin[index], filePath_train_false + '/' + str(k_num))
        k_num += 1
    _, user = os.path.split(user)
    files = get_all_file_path(HP.userdata_path)
    for path in files:
        _, username = os.path.split(path)
        print(path)
        x = get_sensor_data(path)
        target_dir = HP.data_dir + '/' + user + '/test/false'
        if username != user:
            for i, data in enumerate(x):
                # 负样本 采样
                data = sample(data)
                save_data(data, target_dir, str(username) + '-' + str(i), False)


def get_data(path_true, path_false, k_num, multiple, is_test):
    HP = config.HP
    k_num = str(k_num)
    # 获取文件路径
    fileList_true = get_all_file_path(path_true + '/' + k_num)

    if is_test:
        fileList_false = get_all_file_path(path_false)
    else:
        fileList_false = get_all_file_path(path_false + '/' + k_num)

    # 获取时序步长和输入维度
    time_step = HP.time_step

    all_in_feature = HP.all_in_feature
    # 获取数据集
    data_true_acc = get_dir_data(fileList_true, all_in_feature, time_step)
    data_false_acc = get_dir_data(fileList_false, all_in_feature, time_step)

    # 合并传感器数据集

    if not is_test:
        data_true_acc = np.repeat(data_true_acc, multiple, axis=0)
    # 设定输入集
    x_data = np.concatenate((data_true_acc, data_false_acc))
    # 设定输出集
    y_data = np.append(np.ones(np.size(data_true_acc, 0)), (np.zeros(np.size(data_false_acc, 0))))
    # 将输入输出集都转化为张量
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data = torch.tensor(y_data, dtype=torch.float)
    return x_data, y_data


def get_dir_data(fileList_true, in_feature, time_step):
    index = 0
    x_data = np.empty((len(fileList_true), time_step, in_feature), dtype=float)
    for file in fileList_true:
        tmp_data = np.loadtxt(file, delimiter=config.HP.delimiter, encoding='utf-8', dtype=float)
        x_data[index] = tmp_data
        index += 1
    return x_data


def get_sub_data(fileList_true, skiprows, skip_columns, in_feature, time_step):
    index = 0
    x_data = np.empty((len(fileList_true), time_step, in_feature), dtype=float)
    for file in fileList_true:
        tmp_data = np.loadtxt(file, delimiter=config.HP.delimiter, skiprows=skiprows, encoding='utf-8')
        # 出现空的判断一下
        if tmp_data.size == 0:
            tmp_data = np.zeros((time_step - np.size(tmp_data, 0), in_feature))
        # 对数据进行裁剪和填充 形成 time_step*in_feature 维度
        if time_step > np.size(tmp_data, 0):
            tmp_data = np.append(tmp_data, np.zeros((time_step - np.size(tmp_data, 0), in_feature)), 0)
        else:
            tmp_data = tmp_data[:time_step]
        x_data[index] = tmp_data
        index += 1
    x_data = x_data[:, :, skip_columns:in_feature]
    return x_data


def save_data(data, path, name, attach_time):
    if not os.path.exists(path):
        os.makedirs(path)
    if attach_time:
        path += '/' + name + time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime()) + '.txt'
    else:
        path += f'/{name}.txt'
    data = data.to('cpu')

    np.savetxt(path, data, fmt='%.6f', delimiter=' ')


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def save_model(model, optimizer, model_save_path, model_name):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_checkpoint(model, optimizer, os.path.join(model_save_path, model_name))


def get_sensor_data(path):
    fileList_true_acc = get_all_file_path(path + HP.data_dir_acc)
    fileList_true_ori = get_all_file_path(path + HP.data_dir_ori)
    fileList_true_gyro = get_all_file_path(path + HP.data_dir_gyro)
    skiprows = HP.skiprows

    # 获取时序步长和输入维度
    time_step = HP.dataset_time_step
    all_in_feature = HP.all_in_feature
    acc_in_feature = HP.acc_in_feature
    ori_in_feature = HP.ori_in_feature
    gyro_in_feature = HP.gyro_in_feature
    # 获取数据集
    data_acc = get_sub_data(fileList_true_acc, skiprows, 1, acc_in_feature, time_step)
    data_gyro = get_sub_data(fileList_true_gyro, skiprows, 1, gyro_in_feature, time_step)
    data_ori = get_sub_data(fileList_true_ori, skiprows, 1, ori_in_feature, time_step)
    merge_data = np.concatenate((data_acc, data_gyro, data_ori), axis=2)
    return merge_data


if __name__ == '__main__':
    data_init()
