from flask import Flask, render_template, request
import db

class Server:
    def __init__(self, to_server=None, from_server=None) -> None:
        self.to_server = to_server
        self.from_server = from_server

        self.app = Flask(__name__, static_url_path="", static_folder="static")
        
        # register routes
        self.app.add_url_rule("/", view_func=self.homepage)
        self.app.add_url_rule("/new", view_func=self.newpage, methods=["GET", "POST"])


    def homepage(self):
        database = db.get()
        sql_data = database.get_sample()
        sql_new = database.get_new_count()
        sql_new = sql_new[0]
        print(sql_new[0])
        return render_template('index.html', data=sql_data, new=sql_new["count"])

    def newpage(self):
        if request.method == "GET" :
            database = db.get()
            sql_data = database.get_unrecognized_samples()
            sql_new = database.get_new_count()
            sql_new = sql_new[0]
            sql_names = database.get_people()
            return render_template('new.html', data=sql_data, new=sql_new["count"], people=sql_names)

        elif request.method == "POST":
            id = request.form.get("id").split(",")
            fname = request.form.get("fname")
            lname = request.form.get("lname")
            p_id = id[0] # CAREFUL: we need to convert to integer afterwards
            f_id = int(id[1])

            database = db.get()

            # New person
            if p_id == "new":
                p_id = database.insert_person(fname, lname)
                database.add_person_to_image(f_id, p_id)

            else:
                database.add_person_to_image(f_id, int(p_id))

            return "OK"



def run(port=5000, debug=True):
    server = Server()
    server.app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    server = Server()
    run(port=5001)
