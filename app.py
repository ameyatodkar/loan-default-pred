import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('loan_status_predict.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    df = pd.DataFrame([int_features], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                                               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                               'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    prediction = model.predict(df)

    output = "Loan Approved" if prediction == 1 else "Loan Not Approved"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
