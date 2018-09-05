# from __future__ import print_function, division
# from torch.optim import lr_scheduler
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data as Data
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import io
import json
import ast
import sys
import random


BATCH_SIZE = 32
EPOCH = 100
FILE_NUM = 4
OUT_FILE_DIR = "model/"

model_paths = []

def get_batchs(dataset_split_index, inputs, labels):
    x, y = [], []
    b_x, b_y = [], []
    for i, dsi in enumerate(dataset_split_index):
        if (i % BATCH_SIZE) == 0:
            b_x = list_to_cuda_var(b_x)
            b_y = list_to_cuda_var(b_y)
            x.append(b_x)
            y.append(b_y)
            b_x = []
            b_y = []
        rand_split_in_replay = np.random.randint(0, len(inputs[dsi]))
        b_x.append(inputs[dsi][rand_split_in_replay])
        b_y.append(labels[dsi])
    if b_x != []:
        b_x = list_to_cuda_var(b_x)
        b_y = list_to_cuda_var(b_y)
        x.append(b_x)
        y.append(b_y)
    del x[0], y[0]
    return x, y

def list_to_cuda_var(input_list):
    input_list = torch.tensor(input_list)
    input_list = input_list.type(torch.FloatTensor)
    input_list = Variable(input_list.cuda())
    return input_list
    

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

def get_accuracy(predicts, labels):
    hit_num = 0
    if(len(predicts) != len(labels)):
        print("size of predicts and labels didn't match")
        raise ValueError
    else:
        for predict,label in zip(predicts, labels):
            temp = 1 if predict > 0.5 else 0
            if temp == label:
                hit_num += 1
        return hit_num / len(predicts)

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
    model_path = OUT_FILE_DIR + name +'.pt'
    model_paths.append(model_path)
    torch.save(net, model_path)
 
if __name__ == '__main__':
    if not os.path.exists(OUT_FILE_DIR):
        os.mkdir(OUT_FILE_DIR)

    cuda0 = torch.cuda.set_device(0)

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
                
                inputs.append(input_temp)
        label_file_name = feature_file_name + "_label"
        with open(label_file_name) as fi:
            for line in fi.readlines():
                label = ast.literal_eval(line)[-1]
                labels.append(label)
        current_file_num += 1

    if len(labels) != len(inputs):
        print("input length does not match label length!!!")
        sys.exit(-1)
    feature_num = len(inputs[0][0])
    assert feature_num == 99

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
    replay_num = len(inputs)
    replay_index = np.random.permutation(np.arange(replay_num))
    print("replay_num: %d" % replay_num)
    print("replay_index: %s" % str(replay_index))
    train_split_index = int(replay_num * 0.8)
    dev_split_index = int(replay_num * 0.9)
    train_dataset_index = replay_index[ : train_split_index]
    dev_dataset_index = replay_index[train_split_index : dev_split_index]
    test_dataset_index = replay_index[dev_split_index : ]

    #xavier initialization
    linear1 = torch.nn.Linear(feature_num, 256)
    linear2 = torch.nn.Linear(256, 256)
    linear3 = torch.nn.Linear(256, 1)
    # linear3 = torch.nn.Linear(256, 2)
    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    net1 = torch.nn.Sequential(
        linear1,
        torch.nn.Tanh(),
        linear2,
        torch.nn.ReLU(),
        linear3,
        torch.nn.Sigmoid()
    )

    net2, net3 = copy.deepcopy(net1), copy.deepcopy(net1)

    # net1 = torch.load(OUT_FILE_DIR + "net1.pt")
    # net2 = torch.load(OUT_FILE_DIR + "net2.pt")
    # net3 = torch.load(OUT_FILE_DIR + "net3.pt")

    print(net1)  # net 的结构

    # Adam optimizer
    optimizer_1 = torch.optim.Adam(net1.parameters(), lr=1e-2)  # 传入 net 的所有参数, 学习率
    optimizer_2 = torch.optim.Adam(net2.parameters(), lr=1e-3)  # 传入 net 的所有参数, 学习率
    optimizer_3 = torch.optim.Adam(net3.parameters(), lr=1e-4)  # 传入 net 的所有参数, 学习率
    optimizers = [optimizer_1, optimizer_2, optimizer_3]
    loss_func = torch.nn.BCELoss()      # 二分类交叉熵loss
    train_losses_his = [[], [], []]   # 记录 training 时train不同学习率的 loss
    dev_losses_his = [[], [], []]   # 记录 training 时dev不同学习率的 loss
    test_losses_his = [[], [], []]   # 记录 training 时test不同学习率的 loss
    test_acc_his = [[], [], []] # 记录 training 时test不同学习率的 acc
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
        x, y = get_batchs(train_dataset_index, inputs, labels)
        for b_x, b_y in zip(x, y):
            for net, opt, l_his in zip(nets, optimizers, train_losses_his):
                output = net(b_x)              # get output for every net
                output = output.squeeze()
                loss_func.zero_grad()
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                l_his.append(loss.data.cpu().numpy())     # loss recoder

        # dev dataset
        x, y = get_batchs(dev_dataset_index, inputs, labels)
        accuracies = [[], [], []]
        for b_x, b_y in zip(x, y):
            for net, opt, l_his, accuracy in zip(nets, optimizers, dev_losses_his, accuracies):
                output = net(b_x)              # get output for every net
                output = output.squeeze()
                accuracy.append(get_accuracy(output, b_y))
                loss_func.zero_grad()
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                l_his.append(loss.data.cpu().numpy())     # loss recoder
        # compare current accuracy to highest accuracy
        for i in range(len(highest_accuracies)):
            accuracy = np.mean(accuracies[i])
            if accuracy > highest_accuracies[i]:
                highest_accuracies[i] = accuracy
                save_model_for_cpp(nets, i)

        # test dataset
        x, y = get_batchs(test_dataset_index, inputs, labels)
        for b_x, b_y in zip(x, y):
            for net, opt, l_his, acc_his in zip(nets, optimizers, test_losses_his, test_acc_his):
                output = net(b_x)              # get output for every net
                output = output.squeeze()
                acc_his.append(get_accuracy(output, b_y))
                loss_func.zero_grad()
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                l_his.append(loss.data.cpu().numpy())     # loss recoder
    
    # test dataset
    accuracies = [[], [], []]
    x, y = get_batchs(test_dataset_index, inputs, labels)
    for net, model_path, opt, l_his, accuracy in zip(nets, model_paths, optimizers, test_losses_his, accuracies):
        net = torch.load(model_path)
        output = net(b_x)              # get output for every net
        output = output.squeeze()
        accuracy.append(get_accuracy(output, b_y))
        loss = loss_func(output, b_y)
        loss_func.zero_grad()
        l_his.append(loss.data.cpu().numpy())
    test_loss_result = []
    test_accurate_result = []
    for l_his in test_losses_his:
        test_loss_result.append(l_his[-1])
    for accuracy in accuracies:
        test_accurate_result.append(accuracy[-1])
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
    plot_labels = ['lr 1e-2', 'lr 1e-3', 'lr 1e-4']
    
    tlh_plot_data = get_loss_plot_data(train_losses_his)
    dlh_plot_data = get_loss_plot_data(dev_losses_his)
    test_lh_plot_data = get_loss_plot_data(test_losses_his)
    test_ah_plot_data = get_loss_plot_data(test_acc_his)

    # train plot
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
    # test plot
    plt.clf()
    for i, l_his in enumerate(test_lh_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim((0))
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_DIR + "test_loss_summary.png")
    plt.clf()

    plt.clf()
    for i, l_his in enumerate(test_ah_plot_data):
        plt.plot(l_his, label=plot_labels[i])
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accurate')
    plt.xlim((0))
    plt.ylim((0, 1.0))
    plt.savefig(OUT_FILE_DIR + "test_acc_summary.png")
    plt.clf()

    
