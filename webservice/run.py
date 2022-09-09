from flask import Flask, render_template
import db

app = Flask(__name__, static_url_path="", static_folder="static")

@app.route("/")
def run_homepage():
    database = db.get()
    sql_data = database.get_sample()
    print(sql_data)
    return render_template('index.html', data=sql_data)

app.run(host="0.0.0.0", debug=True)
