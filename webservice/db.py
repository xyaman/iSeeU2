import sqlite3 as sql

class DB:
     def __init__(self):
         self.con = sql.connect("RecDB2")
     
     def get_name(self):
         self.cur = self.con.cursor()
         return self.cur.execute("SELECT * FROM Face;")

def get():
    return DB()


# con = sql.connect("RecDB", check_same_thread=False)
# cur = con.cursor()
#  
# def get_name():
#     cur.execute("SELECT path FROM Face;")
