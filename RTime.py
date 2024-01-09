import os
import time

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.utils.data as Data
from ptflops import get_model_complexity_info

import config
import utils
from SelfAttention_Family import AttentionLayer, ProbAttention
from Transformer_Encoder import Encoder, EncoderLayer
from config import HP

seq_len = HP.time_step
d_model = HP.all_in_feature
epochs = HP.epochs
batch_size = HP.batch_size

FACTOR = 3
NUM_CLASS = 1
NUM_LAYER = 1
TOP_K = 4
NUM_KERNELS = 4
WINDOW_SIZE = 7
N_HEADS = 3
ATTN_NUM_LAYER = 1
MULTIPLE = HP.multiple
period_weight = np.zeros([50])

VALUE = HP.Value
LOSS_DECAY = 1.0


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    # 这里对所有batch的频率求平均  代替了对每一个输入数据的频率求前k大的， 就是为了节省时间，前提是同一batch的数据周期具有相似性
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    for i, x in enumerate(period):
        if x < 50:
            period_weight[x] += k - i
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_len, top_k, d_model, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_model, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_model, d_model, num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# features_grad = 0.
# def extract(g):
#     global features_grad
#     features_grad = g
class DTime(nn.Module):
    def __init__(self, num_class, num_layer, seq_len, top_k, d_model, num_kernels, window_size):
        super(DTime, self).__init__()
        time_list = []
        for i in range(num_layer):
            time_list.append(TimesBlock(seq_len, top_k, d_model, num_kernels))
        decomp_list = []
        for i in range(num_layer):
            decomp_list.append(series_decomp(window_size))
        self.time_layer_list = nn.ModuleList(time_list)
        self.decomp_layer_list = nn.ModuleList(decomp_list)
        self.attention = nn.ModuleList(AttentionLayer(ProbAttention(factor=FACTOR), d_model, N_HEADS)
                                       for l in range(ATTN_NUM_LAYER))
        self.projection_1 = nn.Linear(d_model * seq_len * (num_layer + 1), d_model * seq_len * (num_layer + 1) // 2)
        self.projection_2 = nn.Linear(d_model * seq_len * (num_layer + 1) // 2, num_class)
        self.act = F.gelu
        self.layer_norm_1 = nn.LayerNorm(d_model * seq_len * (num_layer + 1) // 2)

        self.trend_list = []
        # self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)

    def forward(self, seasonal_data):
        res = None
        for x, _ in enumerate(self.time_layer_list):
            seasonal_data = self.time_layer_list[x](seasonal_data)
            seasonal_data, tmp_trend_data = self.decomp_layer_list[x](seasonal_data)
            if res is None:
                res = tmp_trend_data
            else:
                res = torch.concatenate([res, tmp_trend_data], dim=0)
        res = torch.concatenate([res, seasonal_data], dim=0)
        res = res.view(NUM_LAYER + 1, seq_len, d_model)
        res = res.permute(1, 0, 2)
        for layer in self.attention:
            res, _ = layer(res, res, res, attn_mask=None)
        res = res.permute(1, 0, 2)
        res = res.reshape(1, -1)
        res = self.projection_1(res)
        res = self.layer_norm_1(res)
        res = self.act(res)
        res = self.projection_2(res)
        res = F.sigmoid(res)
        # res.register_hook(extract)
        # print(features_grad)
        return res, seasonal_data


def evaluate(model, dev_loader, loss_function):
    model.eval()  # 切换验证模式
    sum_loss = 0.
    num_batch = len(dev_loader)
    with torch.no_grad():
        for seq, labels in dev_loader:
            # if numpy.size(seq, 0) < HP.batch_size:
            #     continue
            y_pred, _, _ = model(seq)
            labels = torch.clone(labels.view(HP.batch_size, HP.output_size)).detach().float()
            # for i, j in enumerate(labels):
            #     if j == 0:
            #         if y_pred[i][0] < 0.5:
            #             labels[i][0] = y_pred[i][0]
            #         else:
            #             labels[i][0] = 0.49
            loss = loss_function(y_pred, labels)
            sum_loss += loss.item()

    model.train()  # back to training mode
    return sum_loss / num_batch


def train(userpath, k_num, model_path):
    train_x, train_y = utils.get_data(userpath + '/' + HP.data_dir_train_true,
                                      userpath + '/' + HP.data_dir_train_false, str(k_num), MULTIPLE, False)

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_x, train_y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=HP.batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    model = DTime(NUM_CLASS, NUM_LAYER, seq_len, TOP_K, d_model, NUM_KERNELS, WINDOW_SIZE)
    model.to(HP.device)
    loss_function = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    excel_list = []
    model.train()  # 开始训练
    num_batch = len(train_loader)
    for epoch in range(epochs):
        loss_sum = 0
        time_sum = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            seq = seq.to(HP.device)
            labels = labels.to(HP.device)
            # 将labels转换成-1*output_size的张量
            labels = torch.clone(labels.reshape(-1, HP.output_size)).detach().float()
            if numpy.size(seq, 0) < HP.batch_size:
                continue
            be = time.time()
            y_pred, _ = model(seq)  # 得到输出
            # 如果是负样本，就减少loss的值
            if labels[0] < 1:
                labels[0] = min(VALUE - 0.001, y_pred.detach()[0])
            time_sum += time.time() - be
            loss = loss_function(y_pred, labels)  # 计算损失
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            # print(model.projection_1.weight.grad)
        loss_sum /= num_batch
        print(f"一轮时间：{time_sum / num_batch}")
        print(f"loss:{loss_sum}")
    model_name = 'model_%d.pth' % epochs
    utils.save_model(model, optimizer, model_path, model_name)
    return excel_list


def test(userpath, model_path, k_num):
    k_num = str(k_num)
    prefix, user = os.path.split(userpath)
    test_x, test_y = utils.get_data(userpath + '/' + HP.data_dir_test_true, userpath + '/' + HP.data_dir_test_false,
                                    k_num, 1, True)

    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_x, test_y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=config.HP.batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    num_batch = len(test_loader)
    model = DTime(NUM_CLASS, NUM_LAYER, seq_len, TOP_K, d_model, NUM_KERNELS, WINDOW_SIZE).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    TP = 0.  # 正样本预测答案正确
    FP = 0.  # 错将负样本预测为正样本
    FN = 0.  # 错将正样本标签预测为负样本
    TN = 0.  # 负样本预测答案正确
    season_final = []
    trend_final = []
    origin = []
    with torch.no_grad():
        MAE_sum = 0
        MSE_sum = 0
        for seq, labels in test_loader:
            be = time.time()
            seq = seq.to(HP.device)
            labels = labels.to(HP.device)
            if numpy.size(seq, 0) < HP.batch_size:
                continue
            origin = seq
            y_pred, season_final = model(seq)
            labels = torch.clone(labels.view(-1)).detach().int()
            y_pred = torch.clone(y_pred.view(-1)).detach()
            MAE_sum += utils.MAE(labels, y_pred)
            MSE_sum += utils.MSE(labels, y_pred)
            for i, y in enumerate(y_pred):
                if y >= VALUE:
                    if labels[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if labels[i] == 1:
                        FN += 1
                    else:
                        TN += 1
            print(f'测试时间为:{time.time() - be}')
        print(f'MAE为{MAE_sum / num_batch}')
        print(f'MSE为{MSE_sum / num_batch}')
    # flops, params = get_model_complexity_info(model, input_res=(200, 8), as_strings=True,
    #                                         print_per_layer_stat=True)
    # print("%s %s" % (flops, params))
    if VALUE == 0.5:
        np.savetxt(f'{os.path.split(model_path)[0]}/res.txt', [TP, FP, FN, TN], fmt='%.6f', delimiter=',')

    return TP, FP, FN, TN, MAE_sum / num_batch, MSE_sum / num_batch


def train_all(k_num, is_simple, model_path):
    all_user = utils.get_all_file_path(HP.data_dir)
    writer = pd.ExcelWriter('excel/DTime/train.xlsx', engine='xlsxwriter')
    for user in all_user:
        if is_simple and user != one_pice:
            continue
        print(user)
        _, username = os.path.split(user)
        excel_list = train(user, k_num, os.path.join(model_path, username, 'DTime', str(k_num)))

        pf = pd.DataFrame(excel_list, columns=["轮数", "训练集平均loss值", "验证集平均loss", "保存的模型名称"])
        pf.to_excel(writer, sheet_name=username, index=False)
    writer.close()


def test_all(k_num, is_simple, model_path):
    model_select = [epochs] * 200
    # model_select = [9, 9, 5, 6, 9, 5, 9, 9, 9, 9, 9, 3, 3, 9, 9, 3]
    all_user = utils.get_all_file_path(HP.data_dir)
    data_number = len(all_user)
    Accuracy_sum = 0
    MSE_sum = 0
    Precision_sum = 0
    Recall_sum = 0
    TP_SUM = 0
    FP_SUM = 0
    FN_SUM = 0
    TN_SUM = 0
    for i, user in enumerate(all_user):
        if is_simple and user != one_pice:
            continue
        _, username = os.path.split(user)
        temp = os.path.join(model_path, username, 'DTime', str(k_num), 'model_' + str(model_select[i]) + '.pth')
        TP, FP, FN, TN, MAE, MSE = test(user, temp, k_num)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)  # 准确率
        Precision = -1  # 精确率
        Recall = -1  # 召回率
        print(user)
        print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
        if TP + FP != 0:
            Precision = TP / (TP + FP)
        if TP + FN != 0:
            Recall = TP / (TP + FN)
        print("准确率:", Accuracy * 100.0, '%')
        print("精确率:", Precision * 100.0, '%')
        print("召回率:", Recall * 100.0, '%')
        Accuracy_sum += Accuracy
        MSE_sum += MSE
        Precision_sum += Precision
        Recall_sum += Recall
        TP_SUM += TP
        FP_SUM += FP
        FN_SUM += FN
        TN_SUM += TN
    return Accuracy_sum / data_number, MSE_sum / data_number, Precision_sum / data_number, Recall_sum / data_number, \
           TP_SUM / data_number, FP_SUM / data_number, FN_SUM / data_number, TN_SUM / data_number


# def MSE_reveal(k_num):
#     model_select = [i for i in range(20)]
#     all_user = utils.get_all_file_path(HP.data_dir)
#     MSE_list = [0] * 20
#     for j in range(epochs):
#         for i, user in enumerate(all_user):
#             TP, FP, FN, TN, MAE, MSE = test(user, model_select[j], k_num)
#             MSE_list[j] += MSE
#         MSE_list[j] /= 16
#     np.savetxt('./MSE_list.txt', MSE_list, fmt='%.6f', delimiter=',')

one_pice = './datasets/user1'

if __name__ == '__main__':
    # MSE_reveal()
    for v in range(1, 10):
        ACC_SUM = 0
        MSE_SUM = 0
        Precision_sum = 0
        Recall_sum = 0
        TP_SUM = 0
        FP_SUM = 0
        FN_SUM = 0
        TN_SUM = 0
        is_simple = False
        model_path = 'model_save/uANDnum_exp/u3/num4'
        VALUE = VALUE - 0.01
        for i in range(1, HP.K_NUM + 1):
            # train_all(i, is_simple, model_path)
            ACC, MSE, PRE, REC, TP, FP, FN, TN = test_all(i, is_simple, model_path)
            ACC_SUM += ACC
            MSE_SUM += MSE
            Precision_sum += PRE
            Recall_sum += REC
            TP_SUM += TP
            FP_SUM += FP
            FN_SUM += FN
            TN_SUM += TN
            print(f"{i}则准确率为：{ACC},MSE为：{MSE},"
                  f"Precision为：{PRE},Recall为：{REC}")
        res = [ACC_SUM, MSE_SUM, Precision_sum, Recall_sum, TP_SUM, FP_SUM, FN_SUM, TN_SUM]
        print(f"values:{VALUE}")
        np.savetxt(model_path + f'/roc-{VALUE}.txt', [i / HP.K_NUM for i in res], fmt='%.6f', delimiter=',')
        print(f"k则准确率为：{ACC_SUM / HP.K_NUM},MSE为：{MSE / HP.K_NUM}")
# period_weight = torch.tensor(period_weight, dtype=torch.float)
# x = F.log_softmax(torch.tensor(period_weight))
# np.savetxt('./period_weight.txt', x, fmt='%.6f', delimiter=',')


# def RTime_experiment(top_k, num_layer, num_kernels):
#     is_simple = True
#     for i in range(1, top_k + 1):
#         for j in range(1, num_layer + 1):
#             for nk in range(1, num_kernels + 1):
#                 ACC_SUM = 0
#                 MSE_SUM = 0
#                 Precision_sum = 0
#                 Recall_sum = 0
#                 TP_SUM = 0
#                 FP_SUM = 0
#                 FN_SUM = 0
#                 TN_SUM = 0
#                 global TOP_K, NUM_LAYER, NUM_KERNELS, LOSS_DECAY, WINDOW_SIZE, N_HEADS, FACTOR
#                 TOP_K = i
#                 NUM_LAYER = j
#                 NUM_KERNELS = nk
#                 model_path = f'model_save/RTimesNet_exp'
#                 for k in range(1, HP.K_NUM + 1):
#                     train_all(k, is_simple, model_path)
#                     ACC, MSE, PRE, REC, TP, FP, FN, TN = test_all(k, is_simple, model_path)
#                     ACC_SUM += ACC
#                     MSE_SUM += MSE
#                     Precision_sum += PRE
#                     Recall_sum += REC
#                     TP_SUM += TP
#                     FP_SUM += FP
#                     FN_SUM += FN
#                     TN_SUM += TN
#                 res = [ACC_SUM, MSE_SUM, Precision_sum, Recall_sum, TP_SUM, FP_SUM, FN_SUM, TN_SUM]
#                 np.savetxt(model_path + f'/roc-{i}-{j}-{nk}-{LOSS_DECAY}.txt',
#                            [i / HP.K_NUM for i in res],
#                            fmt='%.6f', delimiter=',')
#                 print(f"k则准确率为：{ACC_SUM / HP.K_NUM},MSE为：{MSE / HP.K_NUM}")
#
#
# if __name__ == '__main__':
#     RTime_experiment(10, 10, 10)
