import sqlite3 as sql

class DB:
    # establishes connection to database
    def __init__(self):
        self.con = sql.connect("RecDB2")
        self.con.row_factory = sql.Row
     
    # returns desired query from desired table
    def get_data(self, table_name, query="*"):
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT {query} FROM {table_name};")
         
        data = table_cur.fetchall()

        return data
    
    # returns time, relative path to photo, name of person
    def get_sample(self):
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT f.time, f.path, p.fname FROM Face f JOIN Person p ON f.person_id=p.id;")

        data = table_cur.fetchall()

        return data
    # returns the count of new unrecognized photos
    def get_new_count(self):
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT COUNT(*) FROM Face WHERE person_id=NULL;")

        data = table_cur.fetchall()

        return data


def get():
    return DB()
