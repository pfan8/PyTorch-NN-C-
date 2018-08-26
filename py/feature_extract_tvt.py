import mysql.connector
import copy
import numpy as np
import time
from enum import Enum
import io

class Race(Enum):
    Zerg = 0
    Terran = 1
    Protoss = 2
    none = 5

def get_max_features(cursor):
    max_features = []
    return max_features

def count_binary_ones(num):
    cnt = 0
    while num:
        if num & 1 == 1:
            cnt += 1
        num = num >> 1
    return cnt


############################    TERRAN    #############################
TERRAN_UNIT_ID_LIST = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 30, 31, 32, 34, 58]
TERRAN_BUILDING_ID_LIST = [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 122, 123, 124, 125]
TERRAN_DEFENSIVE_BUILDINGS = [107, 124, 125]
TERRAN_RESEARCH_BUILDINGS = [108, 112, 122, 120, 123, 115, 116, 117, 118]
##############################    ZERG    #############################
ZERG_UNIT_ID_LIST = [35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 59, 62, 103]
ZERG_BUILDING_ID_LIST = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 149]
ZERG_DEFENSIVE_BUILDINGS = [144, 146]
ZERG_RESEARCH_BUILDINGS = [139, 142, 135, 132, 141, 138, 137, 133, 136, 140]
############################    PROTOSS    #############################
PROTOSS_UNIT_ID_LIST = [60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 83, 84]
PROTOSS_BUILDING_ID_LIST = [154, 155, 156, 157, 159, 160, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172]
PROTOSS_DEFENSIVE_BUILDINGS = [162]
PROTOSS_RESEARCH_BUILDINGS = [159, 163, 164, 165, 166, 169, 170, 171]

########################### DATA #################################
BUILDING_RDO = []
UNIT_SCORE = []

MIN_FRAME_THREASHOLD = 14400
MAX_FRAME_THREASHOLD = 60000
MAX_DISTANCE = 2147483647

# config
DB_NAME = "starcraft_tvt"
HOST = "localhost"
PORT = 3306
USER = "root"
PASSWD = "paofan8"
LIMIT_START_INDEX = 2
LIMIT_NUM = 2
OUTPUT_FILE_NAME = "feature_tvt"
ROW_FILE_NAME = OUTPUT_FILE_NAME + "_row"
def get_features(race, player_replay_ID, replayID, bottom_frame, upper_frame, stats, oppo_stats):

    U_mineral = 0
    U_gas = 0
    U_supply = 0
    query = ("select sum(Minerals),count(Minerals),sum(Gas),count(Gas),sum(Supply),count(Supply),max(TotalMinerals),max(TotalGas),max(TotalSupply) "
                                    "from resourcechange where Frame between %s and %s")
    cursor.execute(query, (bottom_frame, upper_frame))
    sum_m, num_m, sum_g, num_g, sum_s, num_s, mtm, mtg, mts  = cursor.fetchone()
    ##  U_mineral
    if num_m != 0:
        U_mineral = float(sum_m / num_m)
        stats['current_resources'][0] = U_mineral
    else:
        U_mineral = stats['current_resources'][0]
    ##  U_gas
    if num_g != 0:             
        U_gas = float(sum_g / num_g)         
        stats['current_resources'][1] = U_gas
    else:             
        U_gas = stats['current_resources'][1]
    ##  U_supply
    if num_s != 0:             
        U_supply = float(sum_s / num_s)     
        stats['current_resources'][2] = U_supply
    else:            
        U_supply = stats['current_resources'][2]
    ##  I_mineral
    if mtm != 0:             
        I_mineral = float(mtm)
        stats['current_resources'][3] = I_mineral
    else:      
        I_mineral = stats['current_resources'][3]
    ##  I_gas
    if mtg != 0:             
        I_gas = float(mtg)
        stats['current_resources'][4] = I_gas
    else:      
        I_gas = stats['current_resources'][4]
    ##  I_supply
    if mts != 0:             
        I_supply = float(mts)
        stats['current_resources'][5] = I_supply
    else:      
        I_supply = stats['current_resources'][5]

    ##  update current_units and current_buildings
    unit_create_list = []
    unit_destroy_list = []
    query = ("SELECT "
                "EventTypeID,UnitID "
            "FROM "
                "event "
            "WHERE "
                "EventTypeID IN (12,13) "
                    "AND Frame BETWEEN %s AND %s "
                    "AND UnitID IN (SELECT "
                        "UnitID "
                    "FROM "
                        "unit "
                    "WHERE "
                        " PlayerReplayID = %s) ")
    cursor.execute(query, (bottom_frame, upper_frame, player_replay_ID))
    q_result = cursor.fetchall()
    unique_region = 0.0
    if q_result != []:
        for item in q_result:
            etid = item[0]
            uid = item[1]
            query = ("SELECT * FROM unit WHERE UnitID = %s;")
            cursor.execute(query, (uid,))
            unit = cursor.fetchone()
            if etid == 12:
                unit_create_list.append(unit)
            elif etid == 13:
                unit_destroy_list.append(unit)
        for remove_unit in unit_destroy_list:
            if remove_unit in stats['current_buildings']:
                stats['current_buildings'].remove(remove_unit)
            elif remove_unit in stats['current_units']:
                stats['current_units'].remove(remove_unit)
        for add_unit in unit_create_list:
            if add_unit[2] in (TERRAN_UNIT_ID_LIST + ZERG_UNIT_ID_LIST + PROTOSS_UNIT_ID_LIST):
                stats['current_units'].append(add_unit)
            elif add_unit[2] in (TERRAN_BUILDING_ID_LIST + ZERG_BUILDING_ID_LIST + PROTOSS_BUILDING_ID_LIST):
                stats['current_buildings'].append(add_unit)
                ##  unique_region
                for item in BUILDING_RDO:
                    if item[0] == add_unit[2]:
                        unique_region += item[3]
            else:
                continue
                # print("Not Normal Unit/Building !")
    

    ## base_num
    base_num = 0
    for unit in stats['current_buildings']:
        # unit: UnitID, PlayerReplayID, UnitTypeID, UnitReplayID
        # 106: Terran_CommandCenter
        # 131: Zerg_Hatchery
        # 154: Protoss_Nexus
        if unit[2] in (106, 131, 154):
            base_num += 1

    ## building_score
    building_score = 0.0
    for unit in stats['current_buildings']:
        for item in UNIT_SCORE:
            if unit[2] == item[0]:
                building_score += 2*item[1] + 4*item[2]
    
    ## unit_score
    unit_score = 0.0
    for unit in stats['current_units']:
        for item in UNIT_SCORE:
            if unit[2] == item[0]:
                unit_score += 2*item[1] + 4*item[2]
    

    ## building_variety
    building_variety = 0.0
    bt_list = []
    if len(stats['current_buildings']) != 0:
        for building in stats['current_buildings']:
            bt_list.append(building[2])
        building_variety = np.var(bt_list)


    # ## resource_region_num
    # resource_region_num = 0
    # for unit in current_units:
    #     # 110: Terran_Refinery
    #     # 149: Zerg_Extractor
    #     # 157: Protoss_Assimilator
    #     print(unit[2])
    #     if unit[2] in (110, 149, 157):
    #         resource_region_num += 1
    # current_feature.append(resource_region_num)

    ## unit_num 
    unit_num = len(stats['current_units'])

    ## unit_variety 
    unit_variety = 0.0
    ut_list = []
    for unit in stats['current_units']:
        ut_list.append(unit[2])
    if len(stats['current_units']) != 0:
        unit_variety = np.var(ut_list)

    ## vulture_mine
    query = ("SELECT count(*) FROM action WHERE OrderTypeID=132 AND PlayerReplayID=%s "
                    "AND Frame between %s and %s")
    cursor.execute(query, (player_replay_ID, bottom_frame, upper_frame))
    stats['vm_action_num'] += cursor.fetchone()[0]

    ## building_slots
    building_slots = [0] * len(TERRAN_BUILDING_ID_LIST)
    building_total_num = len(stats['current_buildings'])
    if building_total_num != 0:
        for building in stats['current_buildings']:
            for i in range(len(TERRAN_BUILDING_ID_LIST)):
                if building[2] == TERRAN_BUILDING_ID_LIST[i]:
                    building_slots[i] += 1

    ## unit_slots
    unit_slots = [0] * len(TERRAN_UNIT_ID_LIST)
    unit_total_num = len(stats['current_units'])
    if unit_total_num != 0:
        for unit in stats['current_units']:
            for i in range(len(TERRAN_UNIT_ID_LIST)):
                if unit[2] == TERRAN_UNIT_ID_LIST[i]:
                    unit_slots[i] += 1

    feature = [bottom_frame/MAX_FRAME_THREASHOLD, U_mineral, U_gas, U_supply, I_mineral, I_gas, I_supply, base_num, building_score, building_variety, unit_num, unit_score, unit_variety, stats['vm_action_num']
                            , unique_region] + building_slots + unit_slots
    return feature
    

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
start_time = time.time()

cnx = mysql.connector.Connect (host = HOST,
                              port = PORT,
                              user = USER,
                              passwd = PASSWD,
                              db = DB_NAME,
                              buffered=True)
r_cursor = cnx.cursor()
cursor = cnx.cursor()

features = []
max_features = get_max_features(cursor)

cursor.execute ("SELECT * FROM buildingrdo")
BUILDING_RDO = cursor.fetchall()
cursor.execute ("SELECT * FROM unitscore")
UNIT_SCORE = cursor.fetchall()

r_cursor.execute ("SELECT ReplayID FROM replay WHERE ReplayID LIMIT " + str(LIMIT_START_INDEX) + "," + str(LIMIT_NUM))

DEFENSIVE_BUILDINGS = [x for x in BUILDING_RDO if x[1] == 0]
RESEARCH_BUILDINGS = [x for x in BUILDING_RDO if x[1] == 1]

fo_count = 0
# PlayerReplayID,StartPosBTID,Winner,ReplayID
for r_item in r_cursor:
    # get opponent rp_id
    self_rp_id = -1
    oppo_rp_id = -1
    replayID = r_item[0]
    query = "SELECT * FROM playerreplay WHERE ReplayID = " + str(replayID)
    cursor.execute (query)
    for pr_item in cursor:
        player_replay_ID = pr_item[0]
        raceID = pr_item[3]
        if raceID == Race.none.value:
            # print("Invalid Race (PR_ID %d)" % pr_item[0])
            continue
        if self_rp_id == -1:
            self_rp_id = player_replay_ID
        elif oppo_rp_id == -1:
            oppo_rp_id = player_replay_ID
        else:
            print("rpid %d error" % player_replay_ID)
    # filter no Winner flag


    query_max_Frame = ("select Duration from replay where ReplayID=%s")
    cursor.execute(query_max_Frame, (replayID,))
    max_frame = cursor.fetchone()[0]
    print("max_frame: %d" % max_frame)
    if max_frame < MIN_FRAME_THREASHOLD:
        print("Filter it for not reaching min frame threashold")
        continue
    
    print("Processing playreplay %d,%d" % (self_rp_id,oppo_rp_id))
    fo_count += 1
    # output feature to file
    with open(ROW_FILE_NAME,"a") as fo:
        fo.write("row %d playreplay %d,%d" % (fo_count, self_rp_id,oppo_rp_id))
        fo.write("\n")
   
    current_frame_index = 0
    self_current_resources = [0, 0, 0, 0, 0, 0] # resource related,can adapt a different frame interval
    self_vm_action_num = 0
    self_current_units = []
    self_current_buildings = []

    oppo_current_resources = [0, 0, 0, 0, 0, 0]
    oppo_vm_action_num = 0
    oppo_current_units = []
    oppo_current_buildings = []

    self_stats = {'current_resources' : self_current_resources, 
                    'vm_action_num' : self_vm_action_num,
                    'current_units' : self_current_units,
                    'current_buildings' : self_current_buildings}
    oppo_stats = {'current_resources' : oppo_current_resources, 
                    'vm_action_num' : oppo_vm_action_num,
                    'current_units' : oppo_current_units,
                    'current_buildings' : oppo_current_buildings}
    
    ## map_features
    query = ("SELECT MapID, NumStartPos FROM map WHERE MapID in (select MapID from replay where ReplayID = %s)")
    cursor.execute(query, (replayID,))
    mapID, start_pos_num = cursor.fetchone()
    total_tile_num = 0
    gournd_height_0_num = 0
    gournd_height_1_num = 0
    gournd_height_2_num = 0
    gournd_height_3_num = 0
    gournd_height_4_num = 0
    gournd_height_5_num = 0
    buildable_num = 0
    walkable_num = 0
    chokedist_num = 0

    query = ("SELECT GroundHeightID, Buildable, Walkable, ChokeDist FROM buildtile WHERE MapID in (select MapID from replay where ReplayID = %s)")
    cursor.execute(query, (replayID,))
    tiles = cursor.fetchall()
    total_tile_num = len(tiles)
    for GroundHeightID, Buildable, Walkable, ChokeDist in tiles:
        # gournd_height_num
        if GroundHeightID == 0:
            gournd_height_0_num += 1
        elif GroundHeightID == 1:
            gournd_height_1_num += 1
        elif GroundHeightID == 2:
            gournd_height_2_num += 1
        elif GroundHeightID == 3:
            gournd_height_3_num += 1
        elif GroundHeightID == 4:
            gournd_height_4_num += 1
        else:
            gournd_height_5_num += 1
        # buildable_num
        if Buildable:
            buildable_num += 1
        # walkable_num
        walkable_num += count_binary_ones(Walkable)
        # chokedist_num
        if ChokeDist != MAX_DISTANCE:
            chokedist_num += ChokeDist

    gournd_height_0_num /= total_tile_num
    gournd_height_1_num /= total_tile_num
    gournd_height_2_num /= total_tile_num
    gournd_height_3_num /= total_tile_num
    gournd_height_4_num /= total_tile_num
    gournd_height_5_num /= total_tile_num
    buildable_num /= total_tile_num
    walkable_num /= total_tile_num
    chokedist_num /= total_tile_num
    map_features = [gournd_height_0_num, gournd_height_1_num, gournd_height_2_num, gournd_height_3_num, gournd_height_4_num, gournd_height_5_num,
                    buildable_num, walkable_num, chokedist_num]
    
    while current_frame_index * 240 < max_frame:
        bottom_frame = current_frame_index * 240
        current_frame_index += 1
        upper_frame = current_frame_index * 240
        if upper_frame > MAX_FRAME_THREASHOLD:
            break
        # print("current prid: %d" % player_replay_ID)
        # print("current rid: %d" % replayID)
        # print("current frame: %d" % (current_frame_index * 240))
        # print("bottom frame: %d" % (bottom_frame))
        # print("upper frame: %d" % (upper_frame))
        self_current_feature = get_features(Race.Terran, self_rp_id, replayID, bottom_frame, upper_frame, self_stats, oppo_stats)
        opponent_current_feature = get_features(Race.Terran, oppo_rp_id, replayID, bottom_frame, upper_frame, oppo_stats, self_stats)
        # append region values
        self_current_feature.append(self_current_feature[7] - opponent_current_feature[7])
        opponent_current_feature.append(opponent_current_feature[7] - self_current_feature[7])
        ##  concat these features
        features.append(self_current_feature + opponent_current_feature + map_features)
        # print("features: " + str(features))

    # output feature to file
    with open(OUTPUT_FILE_NAME,"a") as fo:
        fo.write(str(features))
        fo.write("\n")
        features.clear()



end_time = time.time()
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("procesasing time is %s" % str(end_time - start_time))
r_cursor.close()
cursor.close()
cnx.close()