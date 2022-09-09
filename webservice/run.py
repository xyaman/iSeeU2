from flask import Flask, render_template
import db

class Server:
    def __init__(self, to_server=None, from_server=None) -> None:
        self.to_server = to_server
        self.from_server = from_server

@app.route("/")
def run_homepage():
    database = db.get()
    sql_data = database.get_sample()
    sql_new = database.get_new_count()
    sql_new = sql_new[0]
    print(sql_new[0])
    return render_template('index.html', data=sql_data, new=sql_new["count"])

@app.route("/new")i
def run_new():
    database = db.get()
    sql_data = database.get_sample()
    sql_new = database.get_new_count()
    sql_new = sql_new[0]
    print(sql_new[0])
    return render_template('new.html', data=sql_data, new=sql_new["count"])

        self.app = Flask(__name__, static_url_path="", static_folder="static")
        
        # register routes
        self.app.add_url_rule("/", view_func=self.homepage)


    def homepage(self):
        database = db.get()
        sql_data = database.get_sample()
        return render_template('index.html', data=sql_data)

    def run(self, port=5000, debug=True):
        self.app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    server = Server()
    server.run()
