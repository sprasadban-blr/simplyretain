import pyhdb as db

def connectDB():
    connection = db.connect("lddbgml.wdf.sap.corp", 30215, "I050385", "Dummy")
    cursor = connection.cursor()
    #cursor.execute("SET SCHEMA SAPABAP1")
    #print(cursor)    
    cursor.execute('SELECT TOP 10 * FROM "SAPABAP1"."RBKP"')
    print(cursor.fetchone())
    cursor.close()
    connection.close()

if __name__ == '__main__':
    connectDB()