###
###   添加建筑类型表-
###   | UnitTypeID | RDOType                   | Addon
###                | 1：研究；0：防御；2：其他  | 1为addon建筑
###
###   
import mysql.connector
from mysql.connector import errorcode

DB_NAME = "starcraft_tvt"
TABLES = {}
TABLES['buildingrdo'] = (
    " CREATE TABLE buildingrdo ("
    " UnitTypeID smallint(6) NOT NULL DEFAULT '228',"
    " RDOType int(11) NOT NULL DEFAULT '-1',"
    " Addon int(11) NOT NULL DEFAULT '-1',"
    " PRIMARY KEY (UnitTypeID),"
    " FOREIGN KEY (UnitTypeID) REFERENCES unittype(UnitTypeID)"
    ") ENGINE=InnoDB")
cnx = mysql.connector.connect (host = "localhost",
                              port = 3306,
                              user = "root",
                              passwd = "paofan8",
                              db = DB_NAME)

cursor = cnx.cursor()
for name, ddl in TABLES.items():
    cursor.execute("DROP TABLE IF EXISTS " + name)
    print("Creating table {}: ".format(name), end='')
    cursor.execute(ddl)
add_building_rdo_type = ("INSERT INTO `buildingrdo` VALUES (108,1,1),(120,1,1),(115,1,1),(117,1,1),(118,1,1),(122,1,0),"
                "(112,1,0),(123,1,0),(116,1,0),(139,1,0),(142,1,0),(135,1,0),"
                "(132,1,0),(141,1,0),(138,1,0),(137,1,0),(133,1,0),(136,1,0),"
                "(140,1,0),(166,1,0),(164,1,0),(163,1,0),(171,1,0),(169,1,0),"
                "(165,1,0),(159,1,0),(170,1,0),"
                "(107,0,1),(125,0,0),(124,0,0),(144,0,0),(146,0,0),(162,0,0),"
                "(109,2,0),(111,2,0),(110,2,0),(113,2,0),(114,2,0),(149,2,0),"
                "(143,2,0),(134,2,0),(157,2,0),(156,2,0),(160,2,0),(172,2,0),"
                "(155,2,0),(167,2,0)")
cursor.execute(add_building_rdo_type)
# Make sure data is committed to the database
cnx.commit()
cursor.close()
cnx.close()