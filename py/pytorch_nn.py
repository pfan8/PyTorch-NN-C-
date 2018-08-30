from __future__ import print_function, division
from torch.optim import lr_scheduler
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import io
import json
import ast
import random


BATCH_SIZE = 32
EPOCH = 10
FILE_NUM = 4
OUT_FILE_PATH = "model/"

def get_loss_plot_data(losses_his):
    temp = []
    result = []
    for his in losses_his:
        his_result = [[]]
        EPOCH_LENGTH = int(len(his) / EPOCH)
        for item in his:
            if len(temp) == EPOCH_LENGTH:
                his_result[0].append(np.mean(temp))
                temp.clear()
            temp.append(item)
        print("his_length: %d" % len(his))
        print("EPOCH_LENGTH: %d" % EPOCH_LENGTH)
        result += his_result
    # print("result: %s" % str(result))
    return result
        

def get_print_tensor(tensor):
    if tensor.dim() < 2:
        return str(tensor.tolist() )
    else:
        t_str = "["
        for i in range(tensor.size()[0]):
            t_str += get_print_tensor(tensor[i]) + ",\n"
        t_str = t_str[:-2]
        t_str += "]"
        return t_str

def normalize_min_max(data, index, max_threashold, min_threashold = 0):
    data[index] = (data[index] - min_threashold) / (max_threashold - min_threashold) if (data[index] < max_threashold) else 1
 
if __name__ == '__main__':
    # torch.cuda.device_count()
    # cuda0 = torch.cuda.set_device(0)
    # torch.cuda.current_device()  # output: 0
    # torch.cuda.get_device_name(0)

    plt.ion() 

    inputs = []
    labels = []
    current_file_num = 0
    while current_file_num < FILE_NUM:
        feature_file_name = "D:/AIIDE/analyzer/replay处理脚本/c++/py/fo_tvt/fo_tvt_" + str(current_file_num * 100) + "_" + str((current_file_num + 1) * 100)
        with open(feature_file_name) as fi:
            while True:
                line = fi.readline()
                if not line:
                    break
                input_temp = ast.literal_eval(line)
                for i in range(len(input_temp)):
                    # U_mineral normalization
                    normalize_min_max(input_temp[i], 1, 4000)
                    # U_gas normalization
                    normalize_min_max(input_temp[i], 2, 1750)
                    # U_supply normalization
                    normalize_min_max(input_temp[i], 3, 350)
                    # I_mineral normalization
                    normalize_min_max(input_temp[i], 4, 75000)
                    # I_gas normalization
                    normalize_min_max(input_temp[i], 5, 38000)
                    # I_supply normalization
                    normalize_min_max(input_temp[i], 6, 400)
                    # base_num normalization
                    normalize_min_max(input_temp[i], 7, 10)
                    normalize_min_max(input_temp[i], 57, 10)
                    # building_score normalization
                    normalize_min_max(input_temp[i], 8, 40000)
                    normalize_min_max(input_temp[i], 58, 40000)
                    # building_variety normalization
                    normalize_min_max(input_temp[i], 9, 60)
                    normalize_min_max(input_temp[i], 59, 60)
                    # unit_num normalization
                    normalize_min_max(input_temp[i], 10, 160)
                    normalize_min_max(input_temp[i], 60, 160)
                    # unit_score normalization
                    normalize_min_max(input_temp[i], 11, 70000)
                    normalize_min_max(input_temp[i], 61, 70000)
                    # unit_variety normalization
                    normalize_min_max(input_temp[i], 12, 30)
                    normalize_min_max(input_temp[i], 62, 30)
                    # vm_action_num normalization
                    normalize_min_max(input_temp[i], 13, 100)
                    normalize_min_max(input_temp[i], 63, 100)
                    # unique_region normalization
                    normalize_min_max(input_temp[i], 14, 30)
                    normalize_min_max(input_temp[i], 64, 30)
                    # building_slots normalization
                    for j in range(15,31):
                        normalize_min_max(input_temp[i], j, 25)
                    for j in range(65,81):
                        normalize_min_max(input_temp[i], j, 25)
                    # unit_slots normalization
                    for j in range(31,49):
                        if j == 31 + 8:
                            normalize_min_max(input_temp[i], j, 120)
                            continue
                        normalize_min_max(input_temp[i], j, 60)
                    for j in range(81,99):
                        if j == 81 + 8:
                            normalize_min_max(input_temp[i], j, 120)
                            continue
                        normalize_min_max(input_temp[i], j, 60)
                    # region_value normalization
                    normalize_min_max(input_temp[i], 49, 20)
                    normalize_min_max(input_temp[i], 99, 20)


                # delete features can not get in real system
                for i in range(len(input_temp)):
                    del input_temp[i][108] # chokedist
                    del input_temp[i][107] # walkable_num (16 binary values)
                    del input_temp[i][64] # oppo unique_region
                    del input_temp[i][51:57] # oppo resourse related features
                    del input_temp[i][14] # self unique_region
                # print(input_temp)
                inputs += input_temp
        label_file_name = feature_file_name + "_label"
        with open(label_file_name) as fi:
            while True:
                line = fi.readline()
                if not line:
                    break
                label = ast.literal_eval(line)
        labels += label
        current_file_num += 1

    # print(inputs[0])
    # print(labels[0])
    # print("feature length: %d" % len(inputs))
    # print("label length: %d" % len(labels))
    # index_array = np.random.permutation(np.arange(len(labels)))
    feature_num = len(inputs[0])
    result_length = len(labels[0]) if type(labels[0]) is list else 1

    # put dateset into torch dataset
    # inputs = torch.Tensor(inputs)
    # print(inputs.size())
    # labels = torch.Tensor(labels)
    # labels = labels.view(-1,1)
    # print(labels.size())
    # dataset = Data.TensorDataset(inputs, labels)

    winner_count = 0
    lose_count = 0
    for item in labels:
        if item == 1:
            winner_count += 1
        elif item == 0:
            lose_count += 1
    print("winner_count: %d" % winner_count)
    print("lose_count: %d" % lose_count)
    # use permutation to split train|dev|test dataset after shuffle whole data
    dataset_index = np.random.permutation(np.arange(len(labels)))
    train_split_index = int(len(inputs) * 0.8)
    dev_split_index = int(len(inputs) * 0.9)
    train_dataset_index = dataset_index[ : train_split_index]
    dev_dataset_index = dataset_index[train_split_index : dev_split_index]
    test_dataset_index = dataset_index[dev_split_index : ]

    #xavier initialization
    linear1 = torch.nn.Linear(feature_num, 256)
    linear2 = torch.nn.Linear(256, 256)
    linear3 = torch.nn.Linear(256, result_length)
    # linear3 = torch.nn.Linear(256, 2)
    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    net = torch.nn.Sequential(
        linear1,
        torch.nn.Tanh(),
        linear2,
        torch.nn.ReLU(),
        linear3,
        torch.nn.Sigmoid()
    )

    net2, net3 = copy.deepcopy(net), copy.deepcopy(net)

    print(net)  # net 的结构

    # Adam optimizer
    optimizer_1 = torch.optim.Adam(net.parameters(), lr=1e-4)  # 传入 net 的所有参数, 学习率
    optimizer_2 = torch.optim.Adam(net2.parameters(), lr=1e-5)  # 传入 net 的所有参数, 学习率
    optimizer_3 = torch.optim.Adam(net3.parameters(), lr=1e-6)  # 传入 net 的所有参数, 学习率
    optimizers = [optimizer_1, optimizer_2, optimizer_3]
    loss_func = torch.nn.CrossEntropyLoss()      # 预测值和真实值的误差计算公式 (交叉熵)
    train_losses_his = [[], [], []]   # 记录 training 时不同学习率的 loss
    dev_losses_his = [[], [], []]   # 记录 training 时不同学习率的 loss
    nets = [net, net2, net3]
    # cuda
    # if torch.cuda.is_available():
    #     net.cuda()
    #     inputs.cuda()
    #     labels.cuda()
    #     loss_func.cuda()
    #     optimizer_1.cuda()
    #     optimizer_2.cuda()
    #     optimizer_3.cuda()


    # training
    since = time.time()
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)

        # train dataset
        bxtemp = [inputs[train_dataset_index[0]]]
        bytemp = [labels[train_dataset_index[0]]]
        for i in range(1, len(train_dataset_index)):
            if i % BATCH_SIZE != 0:
                bxtemp.append(inputs[train_dataset_index[i]])
                bytemp.append(labels[train_dataset_index[i]])
                continue
            b_x, b_y = torch.tensor(bxtemp), torch.tensor(bytemp)
            b_x, b_y = b_x.type(torch.FloatTensor), b_y.type(torch.LongTensor)
            # b_y = b_y.view(-1, 1)
            # b_x, b_y = Variable(b_x.cuda()), Variable(b_y.cuda())
            for net, opt, l_his in zip(nets, optimizers, train_losses_his):
                output = net(b_x)              # get output for every net
                temp = torch.ones(len(output), 1) - output # used for cross entropy loss_func
                output = torch.cat((output,temp), -1)
                # print("output: %s" % str(output))
                # print("b_y: %s" % str(b_y))
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                l_his.append(loss.data.numpy())     # loss recoder
            bxtemp.clear()
            bytemp.clear()

        # dev dataset
        bxtemp = [inputs[dev_dataset_index[0]]]
        bytemp = [labels[dev_dataset_index[0]]]
        for i in range(1, len(dev_dataset_index)):
            if i % BATCH_SIZE != 0:
                bxtemp.append(inputs[dev_dataset_index[i]])
                bytemp.append(labels[dev_dataset_index[i]])
                continue
            b_x, b_y = torch.tensor(bxtemp), torch.tensor(bytemp)
            b_x, b_y = b_x.type(torch.FloatTensor), b_y.type(torch.LongTensor)
            # b_y = b_y.view(-1, 1)
            # b_x, b_y = Variable(b_x.cuda()), Variable(b_y.cuda())
            for net, opt, l_his in zip(nets, optimizers, dev_losses_his):
                output = net(b_x)  
                temp = torch.ones(len(output), 1) - output # used for cross entropy loss_func
                output = torch.cat((output,temp), -1)
                loss = loss_func(output, b_y)
                l_his.append(loss.data.numpy())
            bxtemp.clear()
            bytemp.clear()
    # record time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    plot_labels = ['lr 1e-4', 'lr 1e-5', 'lr 1e-6']
    # train plot
    tlh_plot_data = get_loss_plot_data(train_losses_his)
    dlh_plot_data = get_loss_plot_data(dev_losses_his)
    for i, l_his in enumerate(tlh_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_PATH + "test_train.png")
    plt.clf()
    # seperate optimizer plot
    for i, l_his in enumerate(tlh_plot_data):
        plt.plot(l_his)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim((0, 1.0))
        plt.savefig(OUT_FILE_PATH + plot_labels[i] + "_train.png")
        plt.clf()
    # dev plot
    plt.clf()
    for i, l_his in enumerate(dlh_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_PATH + "test_dev.png")
    plt.clf()
    # seperate optimizer plot
    for i, l_his in enumerate(dlh_plot_data):
        plt.plot(l_his)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim((0, 1.0))
        plt.savefig(OUT_FILE_PATH + plot_labels[i] + "_dev.png")
        plt.clf()
    with open(OUT_FILE_PATH + "model_loss", "w") as fo:
        fo.write("train_loss\n")
        for train_los in train_losses_his:
            fo.write("optimizer\n")
            for item in train_los:
                fo.write(str(item))
                fo.write("\n")
        fo.write("dev_loss\n")
        for dev_loss in dev_losses_his:
            fo.write("optimizer\n")
            for item in dev_loss:
                fo.write(str(item))
                fo.write("\n")

    model_dict = net.state_dict()
    with open(OUT_FILE_PATH + "sc_nn_pytorch.model","w") as fo:
        for i in model_dict:
            if type(model_dict[i]) is torch.Tensor:
                structure = "dict[%s]=" % i
                fo.write(structure + "\n")
                fo.write(str(model_dict[i].size()) + "\n")
                fo.write(get_print_tensor(model_dict[i]))
                fo.write("\n")

    model_dict = net2.state_dict()
    with open(OUT_FILE_PATH + "sc_nn_pytorch.model2","w") as fo:
        for i in model_dict:
            if type(model_dict[i]) is torch.Tensor:
                structure = "dict[%s]=" % i
                fo.write(structure + "\n")
                fo.write(str(model_dict[i].size()) + "\n")
                fo.write(get_print_tensor(model_dict[i]))
                fo.write("\n")

    model_dict = net3.state_dict()
    with open(OUT_FILE_PATH + "sc_nn_pytorch.model3","w") as fo:
        for i in model_dict:
            if type(model_dict[i]) is torch.Tensor:
                structure = "dict[%s]=" % i
                fo.write(structure + "\n")
                fo.write(str(model_dict[i].size()) + "\n")
                fo.write(get_print_tensor(model_dict[i]))
                fo.write("\n")
