import sqlite3 as sql
import os

class DB:
    # establishes connection to database
    def __init__(self):
        self.con = sql.connect(os.path.dirname(__file__) + "/RecDB2")
        self.con.row_factory = sql.Row
     
    # returns desired query from desired table
    def get_data(self, table_name, query="*"):
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT {query} FROM {table_name};")
         
        data = table_cur.fetchall()

        return data
    
    # returns time, relative path to photo, name of person
    def get_sample(self):
        """
        Returns a dictionary with the following fields:
            time: str
            path: str
            fname: str
        """
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT f.time, f.path, p.fname FROM Face f JOIN Person p ON f.person_id=p.id ORDER BY f.time DESC;")

        data = table_cur.fetchall()

        return data

    def get_new_count(self):
        """
            Returns the count of new unrecognized photos
        """
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT COUNT(*) AS count FROM Face WHERE person_id IS NULL;")

        data = table_cur.fetchall()

        return data
    
    def get_unrecognized_samples(self):
        """
        Returns a dictionary with the following fields:
            id: int
            time: str
            path: str
        """

        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT f.id, f.time, f.path FROM Face f WHERE f.person_id IS NULL ORDER BY f.time DESC;")

        data = table_cur.fetchall()

        return data

    def get_people(self):
        """
        Returns a dictionary with the following fields:
            id: int
            fname: str
            lname: str
        """
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"SELECT id, fname, lname FROM Person;")

        data = table_cur.fetchall()

        return data 

    def add_person_to_image(self, image_id, person_id):
        """
        Returns a dictionary with the following fields:
            time: str
            path: str
            fname: str
        """
        self.cur = self.con.cursor()
        table_cur = self.cur.execute(f"UPDATE Face SET person_id={person_id} WHERE id={image_id};")
        self.con.commit()

        data = table_cur.fetchall()

        return data

    def insert_path(self, path):
        """
        Inserts the path of a new photo to the database
        """
        self.cur = self.con.cursor()
        table_cur = self.cur.execute("INSERT INTO Face(path) VALUES('"+path+"');")
        
        self.con.commit()


    def insert_person(self, fname, lname):
        self.cur = self.con.cursor()
        person_cur = self.cur.execute(f"INSERT INTO Person (fname, lname) VALUES ('{fname}', '{lname}')")

        return self.cur.lastrowid

def get():
    return DB()
