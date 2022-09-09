from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html", hello={ "value": "holi" })

@app.route("/new")
def hello_world1():
    return "<p>Hello, new!</p>"

app.run(debug=True)
