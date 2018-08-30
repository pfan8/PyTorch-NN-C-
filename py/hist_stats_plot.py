#hist-plot for frame stats
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
import os

FILE_NUM = 4

def plot_stats(data_stats, file_path):
    png_match = re.match(r'(.*)\.png', file_path)
    if not png_match:
        file_path += ".png"
    data = data_stats['data']
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=100)
    #some stats
    mean = np.mean(data)
    count = len(data)
    dmin = np.min(data)
    dmax = np.max(data)
    dvar = np.var(data)
    text = "avg: {}\ncount: {}\nmin: {}\nmax: {}\nvar:  {}".format(mean, count, dmin, dmax, dvar)
    plt.text(plt.xlim()[1], plt.ylim()[1],text,fontsize=20,ha='right',va='top',bbox=dict(boxstyle='square,pad=0.5',fc='w',ec='k',lw=1))

    plt.title(data_stats['name'] + ' distribution')
    plt.xlabel(data_stats['name'])
    plt.ylabel('num')
    plt.savefig(file_path)
    plt.clf()
    # plt.show()

if __name__ == "__main__":
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
    # stats features
    inputs = np.asarray(inputs)
    U_minerals = np.concatenate((inputs[:,1], inputs[:,51]))
    U_gas = np.concatenate((inputs[:,2], inputs[:,52]))
    U_supply = np.concatenate((inputs[:,3], inputs[:,53]))
    I_mineral = np.concatenate((inputs[:,4], inputs[:,54]))
    I_gas = np.concatenate((inputs[:,5], inputs[:,55]))
    I_supply = np.concatenate((inputs[:,6], inputs[:,56]))
    base_num = np.concatenate((inputs[:,7], inputs[:,57]))
    building_score = np.concatenate((inputs[:,8], inputs[:,58]))
    building_variety = np.concatenate((inputs[:,9], inputs[:,59]))
    unit_num = np.concatenate((inputs[:,10], inputs[:,60]))
    unit_score = np.concatenate((inputs[:,11], inputs[:,61]))
    unit_variety = np.concatenate((inputs[:,12], inputs[:,62]))
    vm_action_num = np.concatenate((inputs[:,13], inputs[:,63]))
    unique_region = np.concatenate((inputs[:,14], inputs[:,64]))
    building_slots = np.concatenate((inputs[:,15:31], inputs[:,65:81]))
    unit_slots = np.concatenate((inputs[:,31:49], inputs[:,81:99]))
    region_value = np.concatenate((inputs[:,49], inputs[:,99]))
    gournd_height_0_num = inputs[:,100]
    gournd_height_1_num = inputs[:,101]
    gournd_height_2_num = inputs[:,102]
    gournd_height_3_num = inputs[:,103]
    gournd_height_4_num = inputs[:,104]
    gournd_height_5_num = inputs[:,105]
    buildable_num = inputs[:,106]
    walkable_num = inputs[:,107]
    chokedist_num = inputs[:,108]

    if not os.path.isdir("stats"):
        os.mkdir("stats")
    # U_minerals
    U_minerals = {
        "name" : "U_minerals",
        "data" : U_minerals
    }
    plot_stats(U_minerals, "stats/U_minerals")
    # U_gas
    U_gas = {
        "name" : "U_gas",
        "data" : U_gas
    }
    plot_stats(U_gas, "stats/U_gas")
    # U_supply
    U_supply = {
        "name" : "U_supply",
        "data" : U_supply
    }
    plot_stats(U_supply, "stats/U_supply")
    # I_mineral
    I_mineral = {
        "name" : "I_mineral",
        "data" : I_mineral
    }
    plot_stats(I_mineral, "stats/I_mineral")
    # I_gas
    I_gas = {
        "name" : "I_gas",
        "data" : I_gas
    }
    plot_stats(I_gas, "stats/I_gas")
    # I_supply
    I_supply = {
        "name" : "I_supply",
        "data" : I_supply
    }
    plot_stats(I_supply, "stats/I_supply")
    # base_num
    base_num = {
        "name" : "base_num",
        "data" : base_num
    }
    plot_stats(base_num, "stats/base_num")
    # building_score
    building_score = {
        "name" : "building_score",
        "data" : building_score
    }
    plot_stats(building_score, "stats/building_score")
    # building_variety
    building_variety = {
        "name" : "building_variety",
        "data" : building_variety
    }
    plot_stats(building_variety, "stats/building_variety")
    # unit_num
    unit_num = {
        "name" : "unit_num",
        "data" : unit_num
    }
    plot_stats(unit_num, "stats/unit_num")
    # unit_score
    unit_score = {
        "name" : "unit_score",
        "data" : unit_score
    }
    plot_stats(unit_score, "stats/unit_score")
    # unit_variety
    unit_variety = {
        "name" : "unit_variety",
        "data" : unit_variety
    }
    plot_stats(unit_variety, "stats/unit_variety")
    # vm_action_num
    vm_action_num = {
        "name" : "vm_action_num",
        "data" : vm_action_num
    }
    plot_stats(vm_action_num, "stats/vm_action_num")
    # unique_region
    unique_region = {
        "name" : "unique_region",
        "data" : unique_region
    }
    plot_stats(unique_region, "stats/unique_region")
    # building_slots
    print(len(building_slots[0]))
    for i in range(len(building_slots[0])):
        name = "building_slot" + str(i)
        building_slot = {
            "name" : name,
            "data" : building_slots[:,i]
        }
        plot_stats(building_slot, "stats/" + name)
    # unit_slots
    for i in range(len(unit_slots[0])):
        name = "unit_slot" + str(i)
        unit_slot = {
            "name" : name,
            "data" : unit_slots[:,i]
        }
        plot_stats(unit_slot, "stats/" + name)
    # region_value
    region_value = {
        "name" : "region_value",
        "data" : region_value
    }
    plot_stats(region_value, "stats/region_value")
    # gournd_height_0_num
    gournd_height_0_num = {
        "name" : "gournd_height_0_num",
        "data" : gournd_height_0_num
    }
    plot_stats(gournd_height_0_num, "stats/gournd_height_0_num")
    # gournd_height_1_num
    gournd_height_1_num = {
        "name" : "gournd_height_1_num",
        "data" : gournd_height_1_num
    }
    plot_stats(gournd_height_1_num, "stats/gournd_height_1_num")
    # gournd_height_2_num
    gournd_height_2_num = {
        "name" : "gournd_height_2_num",
        "data" : gournd_height_2_num
    }
    plot_stats(gournd_height_2_num, "stats/gournd_height_2_num")
    # gournd_height_3_num
    gournd_height_3_num = {
        "name" : "gournd_height_3_num",
        "data" : gournd_height_3_num
    }
    plot_stats(gournd_height_3_num, "stats/gournd_height_3_num")
    # gournd_height_4_num
    gournd_height_4_num = {
        "name" : "gournd_height_4_num",
        "data" : gournd_height_4_num
    }
    plot_stats(gournd_height_4_num, "stats/gournd_height_4_num")
    # gournd_height_5_num
    gournd_height_5_num = {
        "name" : "gournd_height_5_num",
        "data" : gournd_height_5_num
    }
    plot_stats(gournd_height_5_num, "stats/gournd_height_5_num")
    # buildable_num
    buildable_num = {
        "name" : "buildable_num",
        "data" : buildable_num
    }
    plot_stats(buildable_num, "stats/buildable_num")
    # walkable_num
    walkable_num = {
        "name" : "walkable_num",
        "data" : walkable_num
    }
    plot_stats(walkable_num, "stats/walkable_num")
    # chokedist_num
    chokedist_num = {
        "name" : "chokedist_num",
        "data" : chokedist_num
    }
    plot_stats(chokedist_num, "stats/chokedist_num")
