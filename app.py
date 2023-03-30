''' load model and create application with flask'''

import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, url_for
#from django.shortcuts import render


# Load Data
df = pd.read_csv('heart.csv')

# Load Model : teste es in model.pkl umzubenennen 
with open('./model_C=0.01.bin', 'rb') as f_in:
    (sc, lr) = pickle.load(f_in)


# Create Flask Application
app = Flask('heartsick')

# # Homepage
# @app.route('/')
# def home():
#     return render_template('home.html')


# Predict API
@app.route('/predict_api', methods=['POST']) # the adress of the function (endpoint)
def predict_api():

    inp = request.get_json()

    arr = np.array(list(inp.values())).reshape(1, -1)
    patient_t = sc.transform(arr)
    y_pred = lr.predict_proba(patient_t)[0,1]
    doc = bool(y_pred>0.5)

    result = {
        'Probability of Heartsickness': y_pred,
        'Send to Doctor': doc
    }

    # turn patient into json object
    return jsonify(result)

# Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696) # not a local host but 0000 to serve all clients
