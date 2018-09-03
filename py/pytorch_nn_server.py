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
EPOCH = 1000
FILE_NUM = 4
OUT_FILE_DIR = "model_go/"
CLASS_NUM = 2

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
        # print("his_length: %d" % len(his))
        # print("EPOCH_LENGTH: %d" % EPOCH_LENGTH)
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
    data[index] = (data[index] - min_threashold) / (max_threashold - min_threashold)

def get_accuracy(predict, label):
    hit_num = 0
    if(len(predict) != len(label)):
        print("size of predict and label didn't match")
        raise ValueError
    else:
        for p,l in zip(predict, label):
            temp = 1 if p[1] > p[0] else 0
            if temp == l:
                hit_num += 1
        return hit_num / len(predict)

def save_model_for_cpp(nets, i):
    model = {
        "name" : "net" + str(i + 1),
        "net" : nets[i]
    }
    save_model(model)


def save_model(model):
    name = model['name']
    net = model['net']
    model_dict = net.state_dict()
    with open(OUT_FILE_DIR + name + ".model","w") as fo:
        for i in model_dict:
            if type(model_dict[i]) is torch.Tensor:
                structure = "dict[%s]=" % i
                fo.write(structure + "\n")
                fo.write(str(model_dict[i].size()) + "\n")
                fo.write(get_print_tensor(model_dict[i]))
                fo.write("\n")

    torch.save(net, OUT_FILE_DIR + name +'.pt')
 
if __name__ == '__main__':
    if not os.path.exists(OUT_FILE_DIR):
        os.mkdir(OUT_FILE_DIR)

    torch.cuda.device_count()
    cuda0 = torch.cuda.set_device(1)
    torch.cuda.current_device()  # output: 0
    torch.cuda.get_device_name(0)

    plt.ion() 

    inputs = []
    labels = []
    current_file_num = 0
    while current_file_num < FILE_NUM:
        feature_file_name = "fo_tvt/fo_tvt_" + str(current_file_num * 100) + "_" + str((current_file_num + 1) * 100)
        with open(feature_file_name) as fi:
            while True:
                line = fi.readline()
                if not line:
                    break
                input_temp = ast.literal_eval(line)
                for i in range(len(input_temp)):
                    # U_mineral normalization
                    normalize_min_max(input_temp[i], 1, 4000)
                    normalize_min_max(input_temp[i], 51, 4000)
                    # U_gas normalization
                    normalize_min_max(input_temp[i], 2, 1750)
                    normalize_min_max(input_temp[i], 52, 1750)
                    # U_supply normalization
                    normalize_min_max(input_temp[i], 3, 350)
                    normalize_min_max(input_temp[i], 53, 350)
                    # I_mineral normalization
                    normalize_min_max(input_temp[i], 4, 75000)
                    normalize_min_max(input_temp[i], 54, 75000)
                    # I_gas normalization
                    normalize_min_max(input_temp[i], 5, 38000)
                    normalize_min_max(input_temp[i], 55, 38000)
                    # I_supply normalization
                    normalize_min_max(input_temp[i], 6, 400)
                    normalize_min_max(input_temp[i], 56, 400)
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
                    # chokedist normalization
                    normalize_min_max(input_temp[i], 108, 600, 200)

                # delete features can not get from real system
                for i in range(len(input_temp)):
                    del input_temp[i][108] # chokedist
                    del input_temp[i][107] # walkable_num (16 binary values)
                    del input_temp[i][64] # oppo unique_region
                    del input_temp[i][51:57] # oppo resourse related features
                    del input_temp[i][14] # self unique_region
                
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
    np.random.seed(1)
    dataset_index = np.random.permutation(np.arange(len(labels)))
    train_split_index = int(len(inputs) * 0.8)
    dev_split_index = int(len(inputs) * 0.9)
    train_dataset_index = dataset_index[ : train_split_index]
    dev_dataset_index = dataset_index[train_split_index : dev_split_index]
    test_dataset_index = dataset_index[dev_split_index : ]

    #xavier initialization
    linear1 = torch.nn.Linear(feature_num, 256)
    linear2 = torch.nn.Linear(256, 256)
    linear3 = torch.nn.Linear(256, CLASS_NUM)
    # linear3 = torch.nn.Linear(256, 2)
    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    # net1 = torch.nn.Sequential(
    #     linear1,
    #     torch.nn.Tanh(),
    #     linear2,
    #     torch.nn.ReLU(),
    #     linear3
    #     # torch.nn.Sigmoid()
    # )

    # net2, net3 = copy.deepcopy(net1), copy.deepcopy(net1)

    net1 = torch.load(OUT_FILE_DIR + "net1.pt")
    net2 = torch.load(OUT_FILE_DIR + "net2.pt")
    net3 = torch.load(OUT_FILE_DIR + "net3.pt")

    print(net1)  # net 的结构

    # Adam optimizer
    optimizer_1 = torch.optim.Adam(net1.parameters(), lr=1e-3)  # 传入 net 的所有参数, 学习率
    optimizer_2 = torch.optim.Adam(net2.parameters(), lr=9e-3)  # 传入 net 的所有参数, 学习率
    optimizer_3 = torch.optim.Adam(net3.parameters(), lr=1e-4)  # 传入 net 的所有参数, 学习率
    optimizers = [optimizer_1, optimizer_2, optimizer_3]
    loss_func = torch.nn.CrossEntropyLoss()      # 预测值和真实值的误差计算公式 (交叉熵)
    train_losses_his = [[], [], []]   # 记录 training 时不同学习率的 loss
    dev_losses_his = [[], [], []]   # 记录 training 时不同学习率的 loss
    test_losses_his = [[], [], []]   # 记录 training 时不同学习率的 loss
    highest_accuracies = [0, 0, 0]   # 记录最高的准确率
    nets = [net1, net2, net3]

    # cuda
    if torch.cuda.is_available():
        net1.cuda()
        net2.cuda()
        net3.cuda()
        loss_func.cuda()


    # training
    since = time.time()
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        # train dataset
        bxtemp = [inputs[train_dataset_index[0]]]
        bytemp = [labels[train_dataset_index[0]]]
        ac_temp = 0
        for i in range(1, len(train_dataset_index)):
            if i % BATCH_SIZE != 0:
                bxtemp.append(inputs[train_dataset_index[i]])
                bytemp.append(labels[train_dataset_index[i]])
                continue
            b_x, b_y = torch.tensor(bxtemp), torch.tensor(bytemp)
            b_x, b_y = b_x.type(torch.FloatTensor), b_y.type(torch.LongTensor)
            b_x, b_y = Variable(b_x.cuda()), Variable(b_y.cuda())
            for net, opt, l_his in zip(nets, optimizers, train_losses_his):
                output = net(b_x)              # get output for every net
                # temp = Variable(torch.ones(len(output), 1).cuda()) - output # used for cross entropy loss_func
                # output = torch.cat((output,temp), -1)
                loss_func.zero_grad()
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                l_his.append(loss.data.cpu().numpy())     # loss recoder
            bxtemp.clear()
            bytemp.clear()


        # dev dataset
        bxtemp = [inputs[dev_dataset_index[0]]]
        bytemp = [labels[dev_dataset_index[0]]]
        accuracies = [[], [], []]
        for i in range(1, len(dev_dataset_index)):
            if i % BATCH_SIZE != 0:
                bxtemp.append(inputs[dev_dataset_index[i]])
                bytemp.append(labels[dev_dataset_index[i]])
                continue
            b_x, b_y = torch.tensor(bxtemp), torch.tensor(bytemp)
            b_x, b_y = b_x.type(torch.FloatTensor), b_y.type(torch.LongTensor)
            b_x, b_y = Variable(b_x.cuda()), Variable(b_y.cuda())
            for net, opt, l_his, accuracy in zip(nets, optimizers, dev_losses_his, accuracies):
                output = net(b_x)              # get output for every net
                accuracy.append(get_accuracy(output, b_y))
                # temp = Variable(torch.ones(len(output), 1).cuda()) - output # used for cross entropy loss_func
                # output = torch.cat((output,temp), -1)
                loss = loss_func(output, b_y)
                loss_func.zero_grad()
                l_his.append(loss.data.cpu().numpy())
            bxtemp.clear()
            bytemp.clear()
        # compare current accuracy to highest accuracy
        for i in range(len(highest_accuracies)):
            accuracy = np.mean(accuracies[i])
            if accuracy > highest_accuracies[i]:
                highest_accuracies[i] = accuracy
                save_model_for_cpp(nets, i)
    
    # test dataset
    accuracies = [[], [], []]
    for i in range(len(test_dataset_index)):
        bxtemp.append(inputs[test_dataset_index[i]])
        bytemp.append(labels[test_dataset_index[i]])
    b_x, b_y = torch.tensor(bxtemp), torch.tensor(bytemp)
    b_x, b_y = b_x.type(torch.FloatTensor), b_y.type(torch.LongTensor)
    b_x, b_y = Variable(b_x.cuda()), Variable(b_y.cuda())
    for net, opt, l_his, accuracy in zip(nets, optimizers, test_losses_his, accuracies):
        output = net(b_x)              # get output for every net
        accuracy.append(get_accuracy(output, b_y))
        # temp = Variable(torch.ones(len(output), 1).cuda()) - output # used for cross entropy loss_func    
        # output = torch.cat((output,temp), -1)
        loss = loss_func(output, b_y)
        loss_func.zero_grad()
        l_his.append(loss.data.cpu().numpy())
    bxtemp.clear()
    bytemp.clear()
    test_loss_result = []
    test_accurate_result = []
    for l_his in test_losses_his:
        test_loss_result.append(np.mean(l_his))
    for accuracy in accuracies:
        test_accurate_result.append(np.mean(accuracy))
    with open(OUT_FILE_DIR + "test_result.txt", "w") as fo:
        fo.write("loss:\n")
        for result in test_loss_result:
            fo.write(str(result))
            fo.write("\n")
        fo.write("acc:\n")
        for result in test_accurate_result:
            fo.write(str(result))
            fo.write("\n")

    # record time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    plot_labels = ['lr 1e-3', 'lr 9e-3', 'lr 1e-4']
    # train plot
    tlh_plot_data = get_loss_plot_data(train_losses_his)
    dlh_plot_data = get_loss_plot_data(dev_losses_his)
    for i, l_his in enumerate(tlh_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim((0))
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_DIR + "train_summary.png")
    plt.clf()
    # seperate optimizer plot
    for i, l_his in enumerate(tlh_plot_data):
        plt.plot(l_his)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim((0))
        plt.ylim((0, 1.0))
        plt.savefig(OUT_FILE_DIR + plot_labels[i] + "_train.png")
        plt.clf()
    # dev plot
    plt.clf()
    for i, l_his in enumerate(dlh_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim((0))
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_DIR + "dev_summary.png")
    plt.clf()
    # seperate optimizer plot
    for i, l_his in enumerate(dlh_plot_data):
        plt.plot(l_his)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim((0))
        plt.ylim((0, 1.0))
        plt.savefig(OUT_FILE_DIR + plot_labels[i] + "_dev.png")
        plt.clf()
    with open(OUT_FILE_DIR + "train_model_loss", "w") as fo:
        for train_los in train_losses_his:
            fo.write("optimizer\n")
            for item in train_los:
                fo.write(str(item))
                fo.write("\n")
    with open(OUT_FILE_DIR + "dev_model_loss", "w") as fo:
        for dev_loss in dev_losses_his:
            fo.write("optimizer\n")
            for item in dev_loss:
                fo.write(str(item))
                fo.write("\n")

    
