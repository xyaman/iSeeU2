from flask import Flask, render_template
import db

app = Flask(__name__)

@app.route("/")
def run_homepage():
    database = db.get()
    name = database.get_name().fetchall()
    return render_template('index.html', data={"name": name[0]})

app.run(host="0.0.0.0", debug=True)
