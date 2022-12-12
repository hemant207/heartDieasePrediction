import pickle
import numpy as np
import pandas as pd
from flask import Flask, Response,request,app,jsonify,url_for,render_template


app = Flask(__name__)

hear_model = pickle.load(open("random_forest.pkl","rb"))

@app.route('/')
def index():
    return render_template('home.html')

'''
@app.route('/predict_api',method='Post')
def predict_api():
    da = request.json['data']
    print(da)
    return da
'''

@app.route('/abc', methods=['POST'])
def abc():

    req = request.get_json()
    print("hi from abc")
    d = req['data']
    print(d)

    return jsonify(d)


if __name__=="__main__":
    app.run(debug=True)