import mysql.connector
import copy
import numpy as np

def get_max_features(cursor):
    max_features = []
    return max_features

TERRAN_UNIT_ID_LIST = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 30, 31, 32, 34, 58]
TERRAN_BUILDING_ID_LIST = [109, 110, 106, 107, 108, 111, 112, 113, 114, 115, 116, 117, 118, 120, 122, 123, 124, 125]
ZERG_UNIT_ID_LIST = [35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 59, 62, 103]
ZERG_BUILDING_ID_LIST = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 149]
PROTOSS_UNIT_ID_LIST = [60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 83, 84]
PROTOSS_BUILDING_ID_LIST = [154, 155, 156, 157, 159, 160, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172]
BUILDING_RDO = []
DEFENSIVE_BUILDINGS = []
RESEARCH_BUILDINGS = []
UNIT_SCORE = []



cnx = mysql.connector.Connect (host = "localhost",
                              port = 3306,
                              user = "root",
                              passwd = "paofan8",
                              db = "starcraft_tvt",
                              buffered=True)
pr_cursor = cnx.cursor()
cursor = cnx.cursor()

features = []
max_features = get_max_features(cursor)

cursor.execute ("SELECT * FROM buildingrdo")
BUILDING_RDO = cursor.fetchall()
cursor.execute ("SELECT * FROM unitscore")
UNIT_SCORE = cursor.fetchall()

pr_cursor.execute ("SELECT * FROM playerreplay WHERE RaceID != 5"
                  " limit 3")

DEFENSIVE_BUILDINGS = [x for x in BUILDING_RDO if x[1] == 0]
RESEARCH_BUILDINGS = [x for x in BUILDING_RDO if x[1] == 1]

# PlayerReplayID,StartPosBTID,Winner,ReplayID
for pr_item in pr_cursor:
    player_replay_ID = pr_item[0]
    raceID = pr_item[3]
    replayID = pr_item[4]
    start_pos = pr_item[5]
    query_max_Frame = ("select Duration from replay where ReplayID=%s")
    cursor.execute(query_max_Frame, (replayID,))
    max_frame = cursor.fetchone()[0]
    print(max_frame)
    
    current_frame_index = 0
    current_resources = [0, 0, 0, 0, 0, 0]
    vm_action_num = 0
    current_units = []
    current_buildings = []
    
    while current_frame_index * 240 < max_frame:
        current_feature = []
        bottom_frame = current_frame_index * 240
        current_frame_index += 1
        upper_frame = current_frame_index * 240
        print("current prid: %d" % player_replay_ID)
        print("current rid: %d" % replayID)
        print("current frame: %d" % (current_frame_index * 240))
        print("bottom frame: %d" % (bottom_frame))
        print("upper frame: %d" % (upper_frame))
        

        ##  U_mineral
        query = ("select sum(Minerals),count(Minerals) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        sum_r, num_r = cursor.fetchone()
        if num_r != 0:             
            result = float(sum_r / num_r)
            current_resources[0] = result
        else:            
            result = current_resources[0]
        current_feature.append(result)

        ##  U_gas
        query = ("select sum(Gas),count(Gas) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        sum_r, num_r = cursor.fetchone()
        if num_r != 0:             
            result = float(sum_r / num_r)         
            current_resources[1] = result
        else:             
            result = current_resources[1]
        current_feature.append(result)

        ##  U_supply
        query = ("select sum(Supply),count(Supply) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        sum_r, num_r = cursor.fetchone()
        if num_r != 0:             
            result = float(sum_r / num_r)     
            current_resources[2] = result
        else:            
            result = current_resources[2]
        current_feature.append(result)

        ##  I_mineral
        query = ("select max(TotalMinerals) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        num_r = cursor.fetchone()[0]
        if num_r != 0:             
            result = float(num_r)
            current_resources[3] = result
        else:      
            result = current_resources[3]
        current_feature.append(result)

        ##  I_gas
        query = ("select max(TotalGas) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        num_r = cursor.fetchone()[0]
        if num_r != 0:             
            result = float(num_r)      
            current_resources[4] = result
        else:             
            result = current_resources[4]
        current_feature.append(result)

        ##  I_supply
        query = ("select max(TotalSupply) "
                                        "from resourcechange where Frame between %s and %s")
        cursor.execute(query, (bottom_frame, upper_frame))
        num_r = cursor.fetchone()[0]
        if num_r != 0:             
            result = float(num_r)     
            current_resources[5] = result
        else:             
            result = current_resources[5]
        current_feature.append(result)

        ##  update current_units and current_buildings
        unit_create_list = []
        unit_destroy_list = []
        query = ("SELECT "
                    "* "
                "FROM "
                    "starcraft_tvt.event "
                "WHERE "
                    "EventTypeID IN (12,13) "
                        "AND Frame BETWEEN %s AND %s "
                        "AND ReplayID = %s "
                        "AND UnitID IN (SELECT "
                            "UnitID "
                        "FROM "
                            "unit "
                        "WHERE "
                            " PlayerReplayID = %s) ")
        cursor.execute(query, (bottom_frame, upper_frame, replayID, player_replay_ID))
        q_result = cursor.fetchall()
        if q_result != []:
            for item in q_result:
                etid = item[3]
                uid = item[4]
                query = ("SELECT * FROM starcraft_tvt.unit WHERE UnitID = %s;")
                cursor.execute(query, (uid,))
                unit = cursor.fetchone()
                if etid == 12:
                    unit_create_list.append(unit)
                elif etid == 13:
                    unit_destroy_list.append(unit)
            for remove_unit in unit_destroy_list:
                try:
                    current_buildings.remove(remove_unit)
                except Exception as e:
                    print(e)
                try:
                    current_units.remove(remove_unit)
                except Exception as e:
                    print(e)
            for add_unit in unit_create_list:
                if add_unit[2] in (TERRAN_UNIT_ID_LIST + ZERG_UNIT_ID_LIST + PROTOSS_UNIT_ID_LIST):
                    current_units.append(add_unit)
                elif add_unit[2] in (TERRAN_BUILDING_ID_LIST + ZERG_BUILDING_ID_LIST + PROTOSS_BUILDING_ID_LIST):
                    current_buildings.append(add_unit)
                else:
                    print("Not Normal Unit/Building !")
            ##  unique_region
            current_feature.append(len(unit_create_list))
        

        ## base_num
        base_num = 0
        for unit in current_buildings:
            # unit: UnitID, PlayerReplayID, UnitTypeID, UnitReplayID
            # 106: Terran_CommandCenter
            # 131: Zerg_Hatchery
            # 154: Protoss_Nexus
            if unit[2] in (106, 131, 154):
                base_num += 1
        current_feature.append(base_num)

        ## building_score
        current_feature.append(len(current_buildings))

        building_score = 0
        for unit in current_buildings:
            for item in UNIT_SCORE:
                if unit[2] == item[0]:
                    building_score += 2*item[1] + 4*item[2]
        current_feature.append(building_score)

        ## building_variety
        building_variety = 0
        bt_list = []
        for building in current_buildings:
            bt_list.append(building[2])
        if len(current_buildings) != 0:
            building_variety = np.var(bt_list)
        current_feature.append(building_variety)


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

        building_total_num = len(current_buildings)
        ## defensive_ratio
        defensive_ratio = 0
        for building in current_buildings:
            for item in DEFENSIVE_BUILDINGS:
                if building[2] == item[0]:
                    defensive_ratio += 1
        if building_total_num != 0:
            defensive_ratio /= building_total_num
        current_feature.append(defensive_ratio)

        ## research_ratio 
        research_ratio = 0
        for building in current_buildings:
            for item in RESEARCH_BUILDINGS:
                if building[2] == item[0]:
                    defensive_ratio += 1
        if building_total_num != 0:
            research_ratio /= building_total_num
        current_feature.append(research_ratio)

        ## unit_num 
        current_feature.append(len(current_units))

        ## unit_variety 
        unit_variety = 0
        ut_list = []
        for unit in current_units:
            ut_list.append(unit[2])
        if len(current_units) != 0:
            unit_variety = np.var(ut_list)
        current_feature.append(unit_variety)

        ## vulture_mine
        query = ("SELECT * FROM starcraft_tvt.action WHERE OrderTypeID=132 AND PlayerReplayID=%s "
                        "AND Frame between %s and %s")
        cursor.execute(query, (player_replay_ID, bottom_frame, upper_frame))
        q_result = cursor.fetchall()
        if q_result != []:
            vm_action_num += q_result.rowcount
        current_feature.append(vm_action_num)

        ## region_value
        


        ##  concat these features
        features.append(current_feature)
        print("features: " + str(features))



pr_cursor.close()
cursor.close()
cnx.close()