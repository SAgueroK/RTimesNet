# classification config

# 超参配置
# yaml
class Hyperparameter:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cuda'  # cuda
    data_dir = './datasets'
    data_dir_origin_true = 'origin/true'  # 用户的源数据地址
    data_dir_origin_false = 'origin/false'  # 非用户的源数据地址
    data_dir_train_true = 'train/true'  # 用户训练数据地址
    data_dir_train_false = 'train/false'  # 非用户训练数据地址
    data_dir_dev_true = 'dev/true'  # 用户验证数据地址
    data_dir_dev_false = 'dev/false'  # 非用户验证数据地址
    data_dir_test_true = 'test/true'  # 用户测试数据地址
    data_dir_test_false = 'test/false'  # 非用户测试数据地址
    userdata_path = 'raw_data/data_set'  # 用户的源数据地址
    username = 'demo1'
    data_dir_acc = '/acc_data'  # 加速度地址
    data_dir_ori = '/ori_data'  # 倾角地址
    data_dir_gyro = '/gyro_data'  # 角速度地址

    log_dir_eval = './log/eval'  # 日志地址
    log_dir_train = './log/train'  # 日志地址
    model_save_dir = './model_save'   # 模型保存地址

    acc_in_feature = 4  # 线性加速度输入维度
    ori_in_feature = 3  # 倾角输入维度
    gyro_in_feature = 4  # 角速度输入维度
    all_in_feature = 8  # 输入维度
    multiple = 20  # 正样本复制倍数
    time_step = 50  # 时序步
    sample_step = 4  # 抽帧
    dataset_time_step = 200  # 数据有多少帧
    hidden_dim = 8  # 隐含层输出维度
    lstm_layer_num = 10  # LSTM层数
    output_size = 1  # 输出维度
    seed = 1234  # random seed

    # ################################################################
    #                             Model Structure
    # ################################################################
    # ################################################################
    #                             Experiment
    # ################################################################
    batch_size = 1
    init_lr = 1e-3
    epochs = 5
    verbose_step = 10  # 验证频率
    save_step = 20  # 模型保存频率
    shake_threshold = 3  # 抖动阈值
    delimiter = ' '  # 分隔符
    skiprows = 0
    K_NUM = 2
    Value = 0.1
    Value_split = 10
    Value_step = 0.1

HP = Hyperparameter()
