from flask import Flask, render_template, request
import requests

templates = '../templates'
app = Flask(__name__, template_folder=templates)


@app.route("/", methods=['GET'])
@app.route("/index", methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('/index.html')


@app.route('/conversation', methods=['POST'])
def conversation():

    if request.method == 'POST':
        mode = request.form['mode']
        r = requests.post('http://127.0.0.1:5003/talk', data=mode)

        r.raise_for_status()

        if r.json() == 'quit':
            return render_template('/index.html')