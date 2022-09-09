from flask import Flask, render_template
import db

app = Flask(__name__, static_url_path="", static_folder="static")

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


app.run(host="0.0.0.0", debug=True)
