''' load model and create application with flask'''

import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, url_for
#from django.shortcuts import render


# Load Data into df object df
df = pd.read_csv('heart.csv')

# Load objects sc and lr
with open('./model.pkl', 'rb') as f_in:
    sc, lr = pickle.load(f_in)


# Create Flask Application object 
app = Flask(__name__)


#Homepate of Flask App
@app.route('/')
def home():
    return render_template('test.html')


# API Endpoint for prediction from notebook
@app.route('/predict_api', methods=['POST']) 
def predict_api():

    data = request.get_json()
    arr = np.array(list(data.values())).reshape(1, -1)
    patient_t = sc.transform(arr)
    y_pred = lr.predict_proba(patient_t)[0,1] # get positive class
    doc = bool(y_pred > 0.5)
    y_pred = y_pred.round(3)
    result = {
        'Probability of Heartsickness': y_pred,
        'Send to Doctor': doc
    }

    # turn result into json object
    return jsonify(result)



# Homepage with Form to predict heart disease
@app.route('/predict', methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]
    arr = np.array(data).reshape(1, -1)
    patient_t = sc.transform(arr)
    y_pred = lr.predict_proba(patient_t)[0,1] # get positive class
    y_pred = y_pred.round(3)
    return render_template("test.html", prediction_text = f"The Probability of Heart Disease for this Patient is {y_pred}.")


# Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696) # not a local host but 0000 to serve all clients


