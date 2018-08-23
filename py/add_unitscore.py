###
###   添加建筑价位表
###
import mysql.connector
from mysql.connector import errorcode

DB_NAME = "starcraft_tvt"
TABLES = {}
TABLES['unitscore'] = (
    " CREATE TABLE unitscore ("
    " UnitTypeID smallint(6) NOT NULL DEFAULT '228',"
    " Minerals int(11) NOT NULL DEFAULT '-1',"
    " Gas int(11) NOT NULL DEFAULT '-1',"
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
add_unitscore = ("INSERT INTO `unitscore` VALUES (2,75,0),(32,50,25),(1,25,75),(3,100,50),(0,50,0),(34,50,25),(7,50,0)"
                ",(12,400,300),(11,100,100),(14,0,0),(9,100,225),(58,250,125),(8,150,100)"
                ",(68,0,0),(63,0,0),(61,125,100),(66,125,50),(67,50,150),(64,50,0),(83,200,100)"
                ",(85,15,0),(65,100,0),(71,100,350),(72,350,250),(60,150,100),(73,25,0)"
                ",(84,25,75),(70,300,150),(69,200,0),(40,0,0),(46,50,150),(41,50,0),(36,0,0)"
                ",(38,75,25),(50,0,0),(35,0,0),(103,50,100),(97,0,0),(39,200,200),(37,25,0)"
                ",(59,0,0),(62,150,50),(44,50,100),(43,100,100),(42,100,0),(45,100,100)"
                ",(47,12,38),(170,200,150),(157,100,0),(163,150,100),(164,200,0),(169,300,200)"
                ",(166,150,0),(160,150,0),(154,400,0),(159,50,100),(162,150,0),(156,100,0)"
                ",(155,200,200),(171,150,100),(172,100,0),(167,150,150),(165,150,200)"
                ",(143,75,0),(136,150,0),(139,75,0),(149,50,0),(137,100,150),(131,300,0)"
                ",(133,200,150),(135,100,50),(130,0,0),(132,150,100),(134,150,0),(138,150,100)"
                ",(142,200,0),(141,200,150),(144,50,0),(146,50,0),(140,150,200),(112,150,0)"
                ",(123,100,50),(111,150,0),(125,100,0),(106,400,0),(122,125,0),(113,200,100)"
                ",(124,75,0),(110,100,0),(116,100,150),(114,150,100),(109,100,0),(107,50,50)"
                ",(115,50,50),(117,50,50),(120,50,50),(108,100,100),(118,50,50),(5,150,100)"
                ",(13,0,0),(30,0,0)")
cursor.execute(add_unitscore)
# Make sure data is committed to the database
cnx.commit()
cursor.close()
cnx.close()